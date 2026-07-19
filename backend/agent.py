"""
AlgoChat — LLM Agent (ReAct Loop) with Tool Integration
Handles multi-turn conversation with algorithm tool calling.
"""

import json
import logging
from typing import AsyncGenerator, Dict, List, Optional

from config import SYSTEM_PROMPT, MAX_TOOL_STEPS, MODEL_NAME, OPENAI_BASE_URL, OPENAI_API_KEY
from llm_client import get_llm_client
from tools import build_tools, execute_algorithm

logger = logging.getLogger("algochat.agent")


class ChatAgent:
    """LLM-powered agent that can call algorithms as tools.

    Follows a ReAct-style loop:
    1. Send user message + available tools to LLM
    2. If LLM returns text delta → stream to client
    3. If LLM returns tool_call → execute algorithm → send tool_result back
    4. Repeat until LLM stops or max steps reached
    """

    def __init__(self):
        self._llm = get_llm_client()

    def can_run(self) -> bool:
        return bool(OPENAI_API_KEY)

    async def run(
        self,
        user_message: str,
        conversation_history: Optional[List[dict]] = None,
        file_map: Optional[Dict[str, dict]] = None,
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Execute the agent loop with streaming events.

        Yields events: content_delta | tool_call | tool_result | done
        """
        tools = build_tools()
        messages = list(conversation_history) if conversation_history else []

        # ── Build system prompt augmented with file info ──
        system_content = SYSTEM_PROMPT
        if file_map:
            file_descriptions = []
            for fid, finfo in file_map.items():
                name = finfo.get("name", fid)
                file_descriptions.append(
                    f"  • file_id={fid}, name={name}"
                )
            system_content += (
                "\n\n## 当前可用文件\n"
                + "\n".join(file_descriptions)
                + "\n\n调用算法工具时，请将 file_id 作为参数传给工具（如果算法需要文件输入）。"
            )

        # Initialize messages if empty
        if not messages:
            messages.append({"role": "system", "content": system_content})
        elif messages[0].get("role") == "system":
            # Update system prompt with current file info
            messages[0]["content"] = system_content

        messages.append({"role": "user", "content": user_message})

        # ── ReAct loop ──
        step = 0
        all_tool_results: List[dict] = []  # collect all results for done event

        while step < MAX_TOOL_STEPS:
            step += 1
            tool_calls_received: List[dict] = []

            try:
                async for chunk in self._llm.chat_stream(
                    messages=messages,
                    tools=tools if tools else None,
                ):
                    if chunk["type"] == "thinking_delta":
                        yield {"event": "thinking_delta", "content": chunk["content"]}

                    elif chunk["type"] == "delta":
                        yield {"event": "content_delta", "content": chunk["content"]}

                    elif chunk["type"] == "error":
                        # llm_client caught a transport error and yielded a friendly message
                        yield {"event": "error", "message": chunk.get("message", "请求异常，请重试")}

                    elif chunk["type"] == "tool_calls":
                        for tc in chunk["calls"]:
                            tool_calls_received.append(tc)
                            yield {
                                "event": "tool_call",
                                "id": tc.get("id", ""),
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            }

                    elif chunk["type"] == "done":
                        # LLM finished without tool calls → conversation complete
                        if not tool_calls_received:
                            messages.append({
                                "role": "assistant",
                                "content": "",  # content was streamed via deltas
                            })
                            all_results = self._merge_results(all_tool_results)
                            yield {
                                "event": "done",
                                "messages": messages,
                                "results": all_results,
                            }
                            return

                # ── If we received tool calls, execute them ──
                if tool_calls_received:
                    # Record assistant message with tool_calls
                    assistant_msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            }
                            for tc in tool_calls_received
                        ],
                    }
                    messages.append(assistant_msg)

                    # Execute each tool call
                    for tc in tool_calls_received:
                        algo_id = tc["function"]["name"]
                        try:
                            arguments = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}

                        # Inject file paths if file_id is in arguments
                        if file_map:
                            for key, val in arguments.items():
                                if isinstance(val, str) and val.startswith("file_") and val in file_map:
                                    arguments[key] = file_map[val]["path"]

                        logger.info(f"Executing tool: {algo_id} with args: {arguments}")

                        result = execute_algorithm(algo_id, arguments, file_map=file_map)

                        # Yield tool_result event
                        tool_result_data = {
                            "name": algo_id,
                            "success": result.get("success", False),
                            "results": result.get("results", []),
                        }
                        if "error" in result:
                            tool_result_data["error"] = result["error"]

                        yield {"event": "tool_result", **tool_result_data}

                        # Collect results for final aggregation
                        if result.get("results"):
                            all_tool_results.extend(result["results"])

                        # Append tool result message
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": json.dumps(result, ensure_ascii=False),
                        })

                    # Continue loop — LLM will process tool results
                    continue

            except Exception as e:
                logger.error(f"Agent step {step} error: {e}", exc_info=True)
                yield {
                    "event": "error",
                    "message": f"Agent 处理出错 (step {step}): {str(e)}",
                }
                break

        # ── Max steps reached ──
        all_results = self._merge_results(all_tool_results)
        yield {
            "event": "done",
            "messages": messages,
            "results": all_results,
            "max_steps_reached": True,
        }

    def _merge_results(self, results: List[dict]) -> List[dict]:
        """Merge and deduplicate results by id."""
        seen = set()
        merged = []
        for r in results:
            rid = r.get("id", "")
            if rid and rid in seen:
                continue
            seen.add(rid)
            merged.append(r)
        return merged


# Singleton
_agent: Optional[ChatAgent] = None


def get_agent() -> ChatAgent:
    global _agent
    if _agent is None:
        _agent = ChatAgent()
    return _agent