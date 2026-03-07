import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.validation import StreamRequest
from pydantic import ValidationError

from projectdavid.clients.base_client import BaseAPIClient

load_dotenv()
logging_utility = UtilsInterface.LoggingUtility()


class InferenceClient(BaseAPIClient):
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=280.0,
            connect_timeout=10.0,
            read_timeout=280.0,
            write_timeout=30.0,
        )
        self._async_client: Optional[httpx.AsyncClient] = None
        logging_utility.info("InferenceClient initialized using BaseAPIClient.")

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(280.0, connect=10.0),
                headers=(
                    {"X-API-Key": self.api_key} if self.api_key else {}  # ← FIXED
                ),
            )
        return self._async_client

    async def aclose(self):
        if self._async_client:
            await self._async_client.aclose()

    async def create_completion(self, **kwargs) -> Dict[str, Any]:
        final_text = ""
        run_id = kwargs.get("run_id", "unknown")

        async for chunk in self.stream_inference_response(**kwargs):
            final_text += chunk.get("content", "")

        return {
            "id": f"chatcmpl-{run_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": kwargs.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def create_completion_sync(self, **kwargs) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self.create_completion(**kwargs))
                )
                return future.result()
        else:
            return loop.run_until_complete(self.create_completion(**kwargs))

    async def stream_inference_response(
        self,
        model: str,
        thread_id: str,
        message_id: str,
        run_id: str,
        assistant_id: str,
        user_content: Optional[str] = None,
        api_key: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        timeout: float = 600.0,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Asynchronously streams inference.

        CRITICAL: Instantiates a fresh AsyncClient per call — this method is
        frequently invoked from SynchronousInferenceStream running in an
        ephemeral loop, so reusing self.async_client would trigger
        'Future attached to different loop' errors.

        NOTE on api_key vs self.api_key:
        - self.api_key  → platform API key; sent as X-API-Key header
                          so the server can authenticate the caller.
        - api_key param → LLM provider key (Together, OpenAI, Hyperbolic etc.);
                          sent only in the JSON body payload, never in headers.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "api_key": api_key,  # LLM provider key — body only
            "thread_id": thread_id,
            "message_id": message_id,
            "run_id": run_id,
            "assistant_id": assistant_id,
        }

        if user_content:
            payload["content"] = user_content

        if meta_data:
            payload["meta_data"] = meta_data

        try:
            StreamRequest(**payload)
        except ValidationError as e:
            logging_utility.error("Payload validation error: %s", e.json())
            raise

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers=({"X-API-Key": self.api_key} if self.api_key else {}),  # ← FIXED
        ) as client:
            try:
                async with client.stream(
                    "POST", "/v1/completions", json=payload
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue

                        data_str = line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

            except httpx.HTTPStatusError as e:
                logging_utility.error(
                    f"Inference Stream HTTP Error: {e.response.status_code} {e.request.url}"
                )
                raise
            except Exception as e:
                logging_utility.error(f"Inference Stream Unexpected Error: {e}")
                raise

    def close(self):
        if hasattr(self, "client") and self.client:
            self.client.close()
