import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.validation import StreamRequest
from pydantic import ValidationError

# Assuming BaseAPIClient is imported here
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
        """
        Lazily creates and returns a single shared AsyncClient for the MAIN loop.
        """
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(280.0, connect=10.0),
                headers=(
                    {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                ),
            )
        return self._async_client

    async def aclose(self):
        """Cleanly close the async connection pool."""
        if self._async_client:
            await self._async_client.aclose()

    async def create_completion(self, **kwargs) -> Dict[str, Any]:
        """
        Native ASYNC version of completion creation.
        """
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
        """
        Synchronous wrapper with safer loop detection.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import threading
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
        provider: str,
        model: str,
        thread_id: str,
        message_id: str,
        run_id: str,
        assistant_id: str,
        user_content: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Asynchronously streams inference.

        CRITICAL UPDATE: This method now instantiates a fresh AsyncClient if called.
        Because this is often called from 'SynchronousInferenceStream' running in
        a separate thread/loop, we cannot reuse 'self.async_client' (which belongs
        to the main loop) without triggering 'Future attached to different loop' errors.
        """
        payload = {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "thread_id": thread_id,
            "message_id": message_id,
            "run_id": run_id,
            "assistant_id": assistant_id,
        }
        if user_content:
            payload["content"] = user_content

        try:
            # Pydantic validation
            StreamRequest(**payload)
        except ValidationError as e:
            logging_utility.error("Payload validation error: %s", e.json())
            raise

        # Determine headers
        headers = None
        if api_key:
            headers = {"Authorization": f"Bearer {api_key}"}

        # We create a local client context to ensure it binds to the CURRENT loop.
        # This is necessary because SynchronousInferenceStream creates ephemeral loops.
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(280.0, connect=10.0),
            headers=(
                {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            ),
        ) as client:
            try:
                async with client.stream(
                    "POST", "/v1/completions", json=payload, headers=headers
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
                logging_utility.error(f"Inference Stream HTTP Error: {e.response.text}")
                raise
            except Exception as e:
                logging_utility.error(f"Inference Stream Unexpected Error: {e}")
                raise

    def close(self):
        """Closes the underlying synchronous client."""
        # Note: If BaseAPIClient has a .client, close it here.
        if hasattr(self, "client") and self.client:
            self.client.close()
