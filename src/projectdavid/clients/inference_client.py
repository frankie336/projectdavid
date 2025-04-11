import asyncio
import json
import time
from typing import AsyncGenerator, Optional

import httpx
from dotenv import load_dotenv
from httpx import Timeout
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.stream import StreamRequest
from pydantic import ValidationError

load_dotenv()

logging_utility = UtilsInterface.LoggingUtility()

# Define a default timeout configuration
DEFAULT_TIMEOUT = Timeout(30.0, read=60.0)
# Long streaming timeout configuration for cases requiring extended waits
STREAMING_TIMEOUT = Timeout(10.0, read=300.0)  # 5 minutes read timeout


class InferenceClient:
    """
    Client-side service for interacting with the completions endpoint.

    Exposes:
      - create_completion_sync(...): a synchronous wrapper that blocks until the response is aggregated.
      - stream_inference_response(...): an async generator for real-time streaming.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: Optional[Timeout] = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

        # Set up headers once at initialization
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # Apply the timeout configuration to the synchronous client
        self.client = httpx.Client(
            base_url=self.base_url, headers=self.headers, timeout=self.timeout
        )
        logging_utility.info(
            "InferenceClient initialized with base_url: %s and timeout: %s",
            self.base_url,
            self.timeout,
        )

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
        payload = {
            "provider": provider,
            "model": model,
            "thread_id": thread_id,
            "message_id": message_id,
            "run_id": run_id,
            "assistant_id": assistant_id,
        }

        # Apply per-request API key if provided
        if api_key:
            payload["api_key"] = api_key

        if user_content:
            payload["content"] = user_content

        try:
            request = StreamRequest(**payload)
        except ValidationError as e:
            logging_utility.error("Payload validation error: %s", e.json())
            raise ValueError(f"Payload validation error: {e}")

        logging_utility.info(
            "Sending completions request (sync wrapper): %s", request.dict()
        )

        async def aggregate() -> str:
            final_text = ""
            async for chunk in self.stream_inference_response(request):
                final_text += chunk.get("content", "")
            return final_text

        try:
            # Preferred approach: using asyncio.run for clean event loop management
            final_content = asyncio.run(aggregate())
        except RuntimeError as e:
            # Handle cases where this is called from an existing async context
            logging_utility.warning(
                "asyncio.run() detected existing loop or context: %s. Using existing loop.",
                e,
            )
            loop = asyncio.get_event_loop()
            final_content = loop.run_until_complete(aggregate())

        completions_response = {
            "id": f"chatcmpl-{run_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.mapped_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                # Placeholder until proper token counting is implemented
                "prompt_tokens": 0,
                "completion_tokens": len(final_content.split()),
                "total_tokens": len(final_content.split()),
            },
        }
        return completions_response

    async def stream_inference_response(
        self,
        request: StreamRequest,
        streaming_timeout: Optional[Timeout] = None,
    ) -> AsyncGenerator[dict, None]:
        logging_utility.info("Sending streaming inference request: %s", request.dict())

        # Set up headers for this request
        headers = self.headers.copy()

        # Override with request's API key if provided
        if hasattr(request, "api_key") and request.api_key:
            headers["Authorization"] = f"Bearer {request.api_key}"

        # Use streaming timeout if provided, otherwise use the instance timeout
        request_timeout = streaming_timeout if streaming_timeout else self.timeout

        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=request_timeout, headers=headers
        ) as async_client:
            try:
                # Make the streaming request
                async with async_client.stream(
                    "POST",
                    "/v1/completions",
                    json=request.dict(),
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data_str = line[len("data:") :].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                yield chunk
                            except json.JSONDecodeError as json_exc:
                                logging_utility.error(
                                    "Error decoding JSON from stream: %s in line: %s",
                                    str(json_exc),
                                    line,
                                )
                                continue  # Skip malformed lines
            except httpx.TimeoutException as e:
                logging_utility.error("Request timed out: %s", str(e))
                raise
            except httpx.HTTPStatusError as e:
                # Log error details with safer response body handling
                error_summary = (
                    f"Status: {e.response.status_code}, URL: {e.response.url}"
                )
                logging_utility.error(
                    "HTTP error during streaming completions: %s - %s",
                    str(e),
                    error_summary,
                )
                raise
            except httpx.RequestError as e:
                logging_utility.error(
                    "Network or request error during streaming completions: %s", str(e)
                )
                raise
            except Exception as e:
                logging_utility.error(
                    "Unexpected error during streaming completions: %s",
                    str(e),
                    exc_info=True,
                )
                raise

    def close(self):
        """Closes the underlying synchronous HTTPX client."""
        if self.client and not self.client.is_closed:
            self.client.close()
            logging_utility.info("InferenceClient synchronous client closed.")
