"""
AlgoChat — Algorithm-to-OpenAI-Tool Adapter
Converts the @algorithm registry into OpenAI-compatible function/tool definitions.
"""

from typing import List, Dict, Any
from algochat_base import ALGORITHMS, AlgorithmDef


def _build_parameters_schema(algo: AlgorithmDef) -> dict:
    """Build JSON Schema for an algorithm's parameters."""
    properties: Dict[str, dict] = {}
    required: List[str] = []

    for p in algo.params:
        schema: dict = {"description": p.desc or p.label or p.key}

        if p.type == "int":
            schema["type"] = "integer"
            if p.min is not None:
                schema["minimum"] = p.min
            if p.max is not None:
                schema["maximum"] = p.max
            schema["default"] = p.default if p.default is not None else 0
        elif p.type == "float":
            schema["type"] = "number"
            if p.min is not None:
                schema["minimum"] = p.min
            if p.max is not None:
                schema["maximum"] = p.max
            schema["default"] = p.default if p.default is not None else 0.0
        elif p.type == "select":
            schema["type"] = "string"
            if p.options:
                schema["enum"] = p.options
            schema["default"] = p.default if p.default is not None else (p.options[0] if p.options else "")
        elif p.type == "bool":
            schema["type"] = "boolean"
            schema["default"] = p.default if p.default is not None else False
        else:  # text, etc.
            schema["type"] = "string"
            schema["default"] = p.default if p.default is not None else ""

        properties[p.key] = schema
        # Mark as required if no default
        if p.default is None:
            required.append(p.key)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def build_tools() -> List[dict]:
    """Convert all registered algorithms into OpenAI tool definitions."""
    tools = []
    for algo_id, algo in sorted(ALGORITHMS.items()):
        # Build description from algo metadata
        input_desc = ""
        if algo.inputs:
            input_types = ", ".join(algo.inputs[0].types) if algo.inputs else "none"
            input_desc = f" | 需要输入文件类型: {input_types}"

        description = f"{algo.icon} {algo.name} — {algo.desc}{input_desc}"

        # If algorithm has outputs, list them
        if algo.outputs:
            output_names = ", ".join(o.desc or o.key for o in algo.outputs)
            description += f" | 输出: {output_names}"

        tool = {
            "type": "function",
            "function": {
                "name": algo_id,
                "description": description,
                "parameters": _build_parameters_schema(algo),
            },
        }
        tools.append(tool)

    return tools


def execute_algorithm(algo_id: str, arguments: dict, file_map: dict = None) -> dict:
    """
    Execute a registered algorithm by ID with given arguments.
    
    Args:
        algo_id: The algorithm ID to execute
        arguments: Parameters for the algorithm (merged with any file paths)
        file_map: Optional {file_id: {"name": ..., "path": ...}} mapping
    
    Returns formatted results ready for the API response.
    """
    algo = ALGORITHMS.get(algo_id)
    if not algo:
        return {"error": f"算法 '{algo_id}' 不存在", "success": False}

    file_map = file_map or {}

    try:
        # Build inputs dict from algorithm's declared inputs
        inputs = {}
        
        # Collect file paths from arguments that reference uploaded files
        file_paths = []
        for key, val in arguments.items():
            if isinstance(val, str) and val in file_map:
                # Replace file_id with actual path
                arguments[key] = file_map[val]["path"]
                file_paths.append(file_map[val]["path"])
        
        # Map files to algorithm's declared inputs
        if algo.inputs and file_paths:
            first_input = algo.inputs[0]
            if first_input.multiple:
                inputs[first_input.key] = file_paths
            else:
                inputs[first_input.key] = file_paths[:1]
        elif algo.inputs:
            # No files provided — still provide empty input slots
            for inp in algo.inputs:
                inputs[inp.key] = []
        else:
            # Algorithm has no declared inputs — pass any file paths as "data"
            if file_paths:
                inputs["data"] = file_paths
            else:
                inputs["data"] = []

        result = algo.handler(inputs=inputs, params=arguments)

        # Format results using the same logic as main.py
        return _format_results(algo, result)

    except Exception as e:
        return {"error": str(e), "success": False}


def _format_results(algo: AlgorithmDef, result: dict) -> dict:
    """Format algorithm return dict into structured results."""
    outputs = []

    # Dynamic results mode
    dynamic = result.get("_results")
    if isinstance(dynamic, list):
        for i, item in enumerate(dynamic):
            out = {
                "id": f"{algo.id}_dyn_{i}",
                "name": item.get("name", f"结果 {i+1}"),
                "type": item.get("type", "document"),
            }
            if "group" in item:
                out["group"] = item["group"]
            for k in ("chartType", "data", "src", "columns", "rows", "content"):
                if k in item:
                    out[k] = item[k]
            outputs.append(out)
        return {"results": outputs, "success": True}

    # Static results mode
    for out_def in algo.outputs:
        value = result.get(out_def.key)
        if value is None:
            continue

        item = {
            "id": f"{algo.id}_{out_def.key}",
            "name": out_def.desc,
            "type": out_def.type,
        }
        if out_def.group:
            item["group"] = out_def.group

        if out_def.type == "chart":
            if isinstance(value, dict):
                item["chartType"] = value.get("chartType", "bar")
                if "datasets" in value:
                    item["data"] = {"datasets": value["datasets"]}
                    if "labels" in value:
                        item["data"]["labels"] = value["labels"]
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

    return {"results": outputs, "success": True}