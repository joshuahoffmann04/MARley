from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from generator.config import GeneratorRuntimeConfig

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, runtime_config: GeneratorRuntimeConfig) -> None:
        self.config = runtime_config

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        timeout_seconds: float,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        data: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"

        request_obj = urllib.request.Request(
            url=url,
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as response:
                raw_body = response.read()
                if not raw_body:
                    return int(response.status), {}, None
                body = json.loads(raw_body.decode("utf-8"))
                if not isinstance(body, dict):
                    return int(response.status), None, "Response is not a JSON object."
                return int(response.status), body, None
        except urllib.error.HTTPError as exc:
            return int(exc.code), None, str(exc)
        except urllib.error.URLError as exc:
            return None, None, str(exc.reason)
        except (socket.timeout, TimeoutError):
            return None, None, "request timed out"
        except Exception as exc:
            return None, None, str(exc)

    def chat_structured(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        num_ctx: int,
        num_predict: int,
        response_model: Type[T],
    ) -> tuple[T | None, str, str | None]:
        schema_format = response_model.model_json_schema()
        
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": messages,
            "format": schema_format,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }

        status, body, error = self._request_json(
            method="POST",
            url=f"{self.config.ollama_base_url}/api/chat",
            timeout_seconds=self.config.ollama_timeout_seconds,
            payload=payload,
        )

        if status is None or status >= 400 or body is None:
             # Fallback to simple json format if schema fails or not supported?
             # For now, just return error
             return None, "", f"Ollama request failed: {error or status}"

        raw_message = body.get("message", {})
        content = raw_message.get("content", "")
        
        try:
            # Parse JSON from content
            parsed = json.loads(content)
            validated = response_model.model_validate(parsed)
            return validated, content, None
        except Exception as e:
            return None, content, f"Validation failed: {e}"

    def check_readiness(self) -> tuple[bool, dict[str, Any]]:
        status, body, error = self._request_json(
            method="GET",
            url=f"{self.config.ollama_base_url}/api/tags",
            timeout_seconds=min(self.config.ollama_timeout_seconds, 5.0),
        )
        if status and 200 <= status < 300:
            return True, {"available": True, "details": body}
        return False, {"available": False, "error": error}
