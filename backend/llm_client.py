"""
AlgoChat — OpenAI-compatible LLM Client
Supports any OpenAI API-compatible endpoint (OpenAI, DeepSeek, local models, etc.)
"""

import json
import asyncio
import logging
from typing import AsyncGenerator, Dict, List, Optional, Any
import httpx

from config import OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible chat completions client with tool calling support."""

    def __init__(
        self,
        base_url: str = OPENAI_BASE_URL,
        api_key: str = OPENAI_API_KEY,
        model: str = MODEL_NAME,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    @property
    def configured(self) -> bool:
        """Check if API key is set."""
        return bool(self.api_key)

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_request_body(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        stream: bool = False,
    ) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
    ) -> dict:
        """Non-streaming chat completion. Returns the API response message."""
        body = self._build_request_body(messages, tools, stream=False)
        response = await self._http.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]

    async def _post_stream_with_retry(
        self,
        body: dict,
        max_retries: int = 2,
    ):
        """POST to chat/completions with streaming, auto-retry on 422/429."""
        last_status = None
        for attempt in range(max_retries + 1):
            async with self._http.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=body,
            ) as response:
                last_status = response.status_code
                if response.status_code == 200:
                    yield response, None
                    return
                # Read body for error detail (non-streaming error response)
                error_body = await response.aread()
                error_msg = error_body.decode("utf-8", errors="replace")[:500]
                if response.status_code in (422, 429) and attempt < max_retries:
                    wait = 2.0 * (2 ** attempt)  # 2s, 4s
                    await asyncio.sleep(wait)
                    continue
                # Last attempt — raise with body detail
                response._content = error_body  # hack so raise_for_status can read
                response.raise_for_status()
        # Should not reach here, but if all retries exhausted:
        err = httpx.HTTPStatusError(
            f"All {max_retries + 1} attempts failed (last status: {last_status})",
            request=response.request,
            response=response,
        )
        raise err

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Streaming chat completion.
        Yields events: {"type": "delta", "content": "..."} or {"type": "tool_calls", "calls": [...]}
        """
        body = self._build_request_body(messages, tools, stream=True)
        async for response, _err in self._post_stream_with_retry(body):
            tool_calls_accumulator: Dict[int, dict] = {}

            try:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # strip "data: "
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")

                    # Handle reasoning/thinking content (DeepSeek R1, Qwen3, etc.)
                    if delta.get("reasoning_content"):
                        yield {"type": "thinking_delta", "content": delta["reasoning_content"]}

                    # Handle text content deltas
                    if delta.get("content"):
                        yield {"type": "delta", "content": delta["content"]}

                    # Handle tool call deltas (streaming tool calls are accumulated)
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_accumulator:
                                tool_calls_accumulator[idx] = {
                                    "id": tc.get("id", ""),
                                    "function": {"name": "", "arguments": ""},
                                }
                            acc = tool_calls_accumulator[idx]
                            if tc.get("id"):
                                acc["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                acc["function"]["name"] += tc["function"]["name"]
                            if tc.get("function", {}).get("arguments"):
                                acc["function"]["arguments"] += tc["function"]["arguments"]

                    # When streaming finishes and we have tool calls, yield them
                    if finish_reason == "tool_calls" and tool_calls_accumulator:
                        yield {
                            "type": "tool_calls",
                            "calls": [
                                {
                                    "id": v["id"],
                                    "function": {
                                        "name": v["function"]["name"],
                                        "arguments": v["function"]["arguments"],
                                    },
                                }
                                for v in sorted(tool_calls_accumulator.values(), key=lambda x: list(tool_calls_accumulator.keys()))
                            ],
                        }
                        tool_calls_accumulator = {}

                    if finish_reason == "stop":
                        yield {"type": "done"}

            except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
                logger.warning(f"Stream interrupted: {e}")
                yield {"type": "error", "message": "连接中断，请重试"}
                yield {"type": "done"}

    async def close(self):
        await self._http.aclose()


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client