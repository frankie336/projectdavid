import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.validation import StreamRequest
from pydantic import ValidationError

from projectdavid.clients.base_client import BaseAPIClient

ent_validator = ValidationInterface()
load_dotenv()
logging_utility = UtilsInterface.LoggingUtility()


class InferenceClient(BaseAPIClient):
    # Singleton pattern for the async client to maintain a persistent connection pool
    _async_client: Optional[httpx.AsyncClient] = None

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        InferenceClient for interacting with the completions endpoint.
        Inherits BaseAPIClient to maintain unified timeout, auth, and base_url configuration.
        """
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=280.0,
            connect_timeout=10.0,
            read_timeout=280.0,
            write_timeout=30.0,
        )
        logging_utility.info("InferenceClient initialized using BaseAPIClient.")

    def _get_async_client(self) -> httpx.AsyncClient:
        """
        Returns a persistent AsyncClient instance.
        Reusing the client is critical for keeping TCP connections 'warm'.
        """
        if (
            InferenceClient._async_client is None
            or InferenceClient._async_client.is_closed
        ):
            InferenceClient._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    self.timeout, connect=10.0, read=280.0, write=30.0
                ),
                # Limits ensure we don't leak connections while keeping them alive for reuse
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
        return InferenceClient._async_client

    def create_completion_sync(
        self,
        provider: str,
        model: str,
        thread_id: str,
        message_id: str,
        run_id: str,
        assistant_id: str,
        user_content: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """
        Synchronously aggregates the streaming completions result.
        Now benefits from the persistent connection pool in the underlying stream.
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
            # Note: Using .dict() as per current project pattern, model_dump() if pydantic v2
            validated_payload = StreamRequest(**payload)
        except ValidationError as e:
            logging_utility.error("Payload validation error: %s", e.json())
            raise ValueError(f"Payload validation error: {e}")

        async def aggregate() -> str:
            final_text = ""
            async for chunk in self.stream_inference_response(
                provider=provider,
                model=model,
                thread_id=thread_id,
                message_id=message_id,
                run_id=run_id,
                assistant_id=assistant_id,
                user_content=user_content,
                api_key=api_key,
            ):
                final_text += chunk.get("content", "")
            return final_text

        # Using a temporary loop for the sync-to-async aggregation
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            final_content = loop.run_until_complete(aggregate())
        finally:
            loop.close()

        return {
            "id": f"chatcmpl-{run_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(final_content.split()),
                "total_tokens": len(final_content.split()),
            },
        }

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
    ) -> AsyncGenerator[dict, None]:
        """
        Initiates a streaming request using the persistent connection pool.
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
            validated_payload = StreamRequest(**payload)
        except ValidationError as e:
            logging_utility.error("Payload validation error: %s", e.json())
            raise ValueError(f"Payload validation error: {e}")

        # Resolve the client from the persistent pool
        async_client = self._get_async_client()

        # Merge headers for this specific request
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logging_utility.info("Streaming inference request for run_id: %s", run_id)

        try:
            # We no longer 'async with' the client itself, only the stream context
            async with async_client.stream(
                "POST",
                "/v1/completions",
                json=validated_payload.dict(),
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError as json_exc:
                            logging_utility.error(
                                "Error decoding JSON: %s", str(json_exc)
                            )
                            continue
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("Unexpected stream error: %s", str(e))
            raise

    def close(self):
        """
        Closes the underlying synchronous HTTP client.
        Note: The class-level async_client persists for the process lifecycle.
        """
        if hasattr(self, "client") and self.client:
            self.client.close()

    @classmethod
    async def close_async_pool(cls):
        """Cleanly closes the persistent connection pool."""
        if cls._async_client:
            await cls._async_client.aclose()
            cls._async_client = None
