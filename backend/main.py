"""
AlgoChat — Backend Server (FastAPI)
Auto-discovers algorithms from algorithms/ directory via @algorithm decorator
"""

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from typing import Dict, List
import asyncio, json, os, uuid, tempfile, io, logging

from algochat_base import discover_algorithms, ALGORITHMS
from algorithms._utils import generate_fake_data, IMAGE_DIR
from agent import get_agent
from tools import build_tools, execute_algorithm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("algochat")

app = FastAPI(title="AlgoChat Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Auto-discover algorithms on startup ──
discover_algorithms()

# ── In-memory conversation store (keyed by conversation_id) ──
# Each conversation holds the full OpenAI-compatible messages list:
#   [system, user, assistant, tool, user, assistant, …]
_conversations: dict[str, list[dict]] = {}
_conversations_lock = asyncio.Lock()


# ── API Routes ──

@app.get("/api/algorithms")
async def list_algorithms():
    """Return all registered algorithms with full schema (inputs/outputs/params)."""
    return [algo.to_dict() for algo in ALGORITHMS.values()]


@app.post("/api/chat")
async def chat(message: str = Form(...), conversation_id: str = Form(None)):
    """Simple keyword-based chat routing."""
    msg = message.lower()
    keyword_map = {
        "聚类": "kmeans", "kmeans": "kmeans",
        "dbscan": "dbscan",
        "回归": "linear_regression", "线性": "linear_regression",
        "降维": "pca", "pca": "pca",
        "异常": "anomaly_detection", "检测": "anomaly_detection",
        "统计": "data_stats", "分析": "data_stats", "图表": "data_stats", "可视化": "data_stats", "数据": "data_stats",
        "图片": "image_batch", "批处理": "image_batch", "批量": "image_batch", "图像": "image_batch",
    }
    for kw, algo_id in keyword_map.items():
        if kw in msg:
            algo = ALGORITHMS.get(algo_id)
            if algo:
                result = algo.handler(inputs={}, params={})
                results_list = _format_results(algo, result)
                return {"type": "results", "message": f"**{algo.name}**执行完成", "results": results_list}
    return {"type": "chat", "message": f"收到消息: {message}\n\n试试关键词：聚类、回归、降维、异常、统计\n或在左侧选择算法上传文件执行。"}


@app.post("/api/algorithm/run")
async def run_algorithm(algorithm: str = Form(...), params: str = Form("{}"), files: list[UploadFile] = File(default=[])):
    """Run a specific algorithm with uploaded files and parameters."""
    algo = ALGORITHMS.get(algorithm)
    if not algo:
        return {"type": "error", "message": f"未知算法: {algorithm}"}

    parsed_params = json.loads(params)

    # Save uploaded files to temp paths, grouped by input key
    saved_files = []
    for f in files:
        path = os.path.join(tempfile.gettempdir(), f"algochat_{uuid.uuid4().hex[:8]}_{f.filename}")
        with open(path, "wb") as out:
            out.write(await f.read())
        saved_files.append(path)

    # Build inputs dict — map files to the first input key
    inputs = {}
    if algo.inputs:
        first_input = algo.inputs[0]
        if first_input.multiple:
            inputs[first_input.key] = saved_files
        else:
            inputs[first_input.key] = saved_files[:1]  # single file as list
    else:
        inputs["data"] = saved_files

    try:
        result = algo.handler(inputs=inputs, params=parsed_params)
        results_list = _format_results(algo, result)
        return {"type": "results", "message": f"**{algo.name}**执行完成", "results": results_list}
    except Exception as e:
        return {"type": "error", "message": str(e)}


def _format_results(algo_def, result: dict) -> list:
    """Format algorithm return dict into the API results list.

    The algorithm handler returns {output_key: value}.
    We match each key to the declared Output to determine type and name.
    
    If the handler returns a list of dicts instead, each dict must contain
    'key' + type-specific fields + optional 'group'. This enables dynamic
    results whose count is determined at runtime (e.g. per-input-file).
    """
    outputs = []

    # ── Dynamic results mode ──
    # If result contains a special "_results" key with a list, use that directly.
    # This allows algorithms to return a variable number of results at runtime.
    dynamic = result.get("_results")
    if isinstance(dynamic, list):
        for i, item in enumerate(dynamic):
            out = {
                "id": f"{algo_def.id}_dyn_{i}",
                "name": item.get("name", f"结果 {i+1}"),
                "type": item.get("type", "document"),
            }
            # Copy group if present
            if "group" in item:
                out["group"] = item["group"]
            # Copy type-specific fields
            for k in ("chartType", "data", "src", "columns", "rows", "content"):
                if k in item:
                    out[k] = item[k]
            outputs.append(out)
        return outputs

    # ── Static results mode (original) ──
    for out_def in algo_def.outputs:
        value = result.get(out_def.key)
        if value is None:
            continue

        item = {"id": f"{algo_def.id}_{out_def.key}", "name": out_def.desc, "type": out_def.type}
        # Pass group from Output definition
        if out_def.group:
            item["group"] = out_def.group

        if out_def.type == "chart":
            if isinstance(value, dict):
                item["chartType"] = value.get("chartType", "bar")
                if "datasets" in value:
                    item["data"] = {"datasets": value["datasets"]}
                    if "labels" in value:
                        item["data"]["labels"] = value["labels"]
                # Allow dynamic group from value
                if "group" in value:
                    item["group"] = value["group"]
        elif out_def.type == "image":
            if isinstance(value, dict):
                item["src"] = value.get("src", "")
                item["name"] = value.get("name", out_def.desc)
                if "group" in value:
                    item["group"] = value["group"]
        elif out_def.type == "table":
            if isinstance(value, dict):
                item["columns"] = value.get("columns", [])
                item["rows"] = value.get("rows", [])
                if "group" in value:
                    item["group"] = value["group"]
        elif out_def.type == "document":
            if isinstance(value, str):
                item["content"] = value
            elif isinstance(value, dict):
                item["content"] = value.get("content", str(value))
                if "group" in value:
                    item["group"] = value["group"]

        outputs.append(item)
    return outputs


@app.get("/api/data/generate")
async def generate_data(dataset: str = Query("clusters"), n_samples: int = Query(200)):
    try:
        df = generate_fake_data(dataset, n_samples)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 400)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(io.BytesIO(output.getvalue().encode("utf-8-sig")), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=fake_{dataset}_{n_samples}.csv"})


@app.get("/api/images/{filename}")
async def get_image(filename: str):
    path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return JSONResponse({"error": "图片未找到"}, 404)


@app.get("/api/files/{file_id}/download")
async def download_file(file_id: str):
    path = os.path.join(tempfile.gettempdir(), file_id)
    if os.path.exists(path):
        return FileResponse(path, filename=os.path.basename(path))
    return JSONResponse({"error": "文件未找到"}, 404)


# ═══════════════════════════════════════════════════════
# Agent Chat SSE Endpoint (v4 — LLM-powered)
# ═══════════════════════════════════════════════════════

# Per-conversation file mapping: conv_id → {file_id: file_path}
_conv_files: Dict[str, Dict[str, str]] = {}
_conv_files_lock = asyncio.Lock()


@app.post("/api/chat/agent")
async def agent_chat(
    message: str = Form(...),
    conversation_id: str = Form(None),
    files: List[UploadFile] = File(None),
    prompt: str = Form(None),  # optional pre-built prompt from frontend (e.g. "run kmeans on this data")
):
    """LLM Agent chat with streaming SSE response.

    The agent has access to all registered algorithms as tools.
    It streams: content_delta → tool_call → tool_result → content_delta → ... → done

    Maintains per-conversation message history in openai-compatible format
    so that the LLM sees full context across multiple turns.
    """
    agent = get_agent()

    if not agent.can_run():
        async def no_config_generator():
            yield {
                "event": "error",
                "data": json.dumps({"message": "LLM 未配置。请在 .env 中设置 OPENAI_API_KEY。"}),
            }
        return EventSourceResponse(no_config_generator())

    # ── Resolve conversation_id ──
    cid = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"

    # ── Save uploaded files ──
    file_map: Dict[str, str] = {}
    if files:
        async with _conv_files_lock:
            if cid not in _conv_files:
                _conv_files[cid] = {}
        for f in files:
            if not f.filename:
                continue
            # Sanitize filename
            safe_name = f.filename.replace("\\", "_").replace("/", "_")
            upload_dir = os.path.join(tempfile.gettempdir(), "algochat_uploads", cid)
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, safe_name)
            content = await f.read()
            with open(file_path, "wb") as fp:
                fp.write(content)
            file_id = f"file_{uuid.uuid4().hex[:8]}"
            file_map[file_id] = {"name": safe_name, "path": file_path}
            async with _conv_files_lock:
                _conv_files[cid][file_id] = {"name": safe_name, "path": file_path}

    # ── Build effective user message ──
    effective_message = message
    if file_map:
        file_list = "\n".join(f"- {v['name']}" for v in file_map.values())
        effective_message = f"已上传文件:\n{file_list}\n\n用户消息: {message}"

    # ── Retrieve conversation history (openai-compatible messages list) ──
    history = []
    if conversation_id and conversation_id in _conversations:
        history = _conversations[conversation_id]

    async def event_generator():
        try:
            async for event in agent.run(
                user_message=effective_message,
                conversation_history=history,
                file_map=file_map,
                conversation_id=cid,
            ):
                yield {"event": event["event"], "data": json.dumps(
                    {k: v for k, v in event.items() if k != "event"},
                    ensure_ascii=False,
                    default=str,
                )}

                # After the final 'done' event, save updated messages back
                if event.get("event") == "done":
                    # Also tell the frontend which conversation_id to use
                    yield {
                        "event": "conversation_id",
                        "data": json.dumps({"conversation_id": cid}),
                    }
                    messages = event.get("messages", [])
                    async with _conversations_lock:
                        _conversations[cid] = list(messages)
        except Exception as e:
            logger.error(f"Agent chat error: {e}", exc_info=True)
            # Provide user-friendly Chinese error messages
            error_str = str(e)
            status_code = None
            try:
                status_code = e.response.status_code
            except Exception:
                pass
            if status_code == 422:
                friendly_msg = "模型暂时无法处理该请求（422），请稍后重试或简化输入内容。"
            elif status_code == 429:
                friendly_msg = "请求过于频繁（429），请稍后重试。"
            elif status_code == 401:
                friendly_msg = "API 密钥无效或已过期，请在设置中更新。"
            elif status_code == 503:
                friendly_msg = "模型服务暂时不可用（503），请稍后重试。"
            else:
                friendly_msg = f"处理出错: {error_str}"
            yield {
                "event": "error",
                "data": json.dumps({"message": friendly_msg}),
            }

    return EventSourceResponse(event_generator())


@app.get("/api/chat/config")
async def chat_config():
    """Return agent configuration status for the frontend."""
    agent = get_agent()
    return {
        "llm_configured": agent.can_run(),
        "tools_count": len(build_tools()),
        "model": agent._llm.model if agent.can_run() else None,
    }


# ═══════════════════════════════════════════════════════
# Static file serving — single entry point (open browser, get the UI)
# ═══════════════════════════════════════════════════════

STATIC_ROOT = os.path.join(os.path.dirname(__file__), "..")

app.mount("/css", StaticFiles(directory=os.path.join(STATIC_ROOT, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(STATIC_ROOT, "js")), name="js")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_ROOT, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
