import asyncio
import json
import time
from typing import AsyncGenerator, Optional

import httpx
from dotenv import load_dotenv

# Import Timeout configuration object
from httpx import Timeout
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.stream import StreamRequest
from pydantic import ValidationError

load_dotenv()

logging_utility = UtilsInterface.LoggingUtility()

# Define a default timeout configuration (e.g., 10s connect, 60s read/write/pool)
# You can adjust these values as needed.
DEFAULT_TIMEOUT = Timeout(30.0, read=60.0)


# Alternatively, for very long streaming waits, you might need a much longer read timeout:
# DEFAULT_TIMEOUT = Timeout(10.0, read=300.0) # 5 minutes read timeout


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
        self.timeout = (
            timeout if timeout is not None else DEFAULT_TIMEOUT
        )  # Use provided or default timeout

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Apply the timeout configuration to the synchronous client
        self.client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=self.timeout
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
        # Note: This api_key overrides the instance one for the request
    ) -> dict:
        payload = {
            "provider": provider,
            "model": model,
            # If api_key is provided here, it should ideally be used for the request,
            # but the current stream_inference_response uses the instance self.api_key.
            # Consider how you want to handle potentially different API keys.
            # For simplicity, we'll stick to the instance key for now.
            "api_key": api_key,
            "thread_id": thread_id,
            "message_id": message_id,
            "run_id": run_id,
            "assistant_id": assistant_id,
        }
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
            # Pass the request object, stream_inference_response handles the rest
            async for chunk in self.stream_inference_response(request):
                final_text += chunk.get("content", "")
            return final_text

        # Using asyncio.run is generally preferred over manually managing event loops
        try:
            final_content = asyncio.run(aggregate())
        except RuntimeError as e:
            # Handle cases where asyncio.run detects an existing running loop
            # (e.g., if called from within another async context like FastAPI)
            logging_utility.warning(
                "asyncio.run() detected existing loop or context: %s. Using existing loop.",
                e,
            )
            # This part might need adjustment based on the exact context where create_completion_sync is called
            # If it's always called from a sync context, the original new_event_loop might be okay,
            # but asyncio.run is more robust. If called from async, it shouldn't use run_until_complete directly.
            # For now, let's assume it might run into this issue and log it.
            # A more robust solution might involve checking if a loop is running first.
            loop = asyncio.get_event_loop()
            final_content = loop.run_until_complete(aggregate())

        completions_response = {
            "id": f"chatcmpl-{run_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.mapped_model,  # Use mapped_model from the validated request
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
                # Note: Calculating tokens accurately usually requires a tokenizer
                "prompt_tokens": 0,  # Placeholder
                "completion_tokens": len(final_content.split()),  # Rough estimate
                "total_tokens": len(final_content.split()),  # Rough estimate
            },
        }
        return completions_response

    async def stream_inference_response(
        self,
        request: StreamRequest,
    ) -> AsyncGenerator[dict, None]:
        logging_utility.info("Sending streaming inference request: %s", request.dict())

        # Prepare headers for the async client
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # If request.api_key exists and should override instance key, handle it here:
        # if request.api_key:
        #    headers["Authorization"] = f"Bearer {request.api_key}"

        # Create the async client with the configured timeout and headers
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout, headers=headers
        ) as async_client:
            # No need to set headers again here if passed during client creation
            # if self.api_key:
            #     async_client.headers["Authorization"] = f"Bearer {self.api_key}"

            try:
                # Make the streaming request
                async with async_client.stream(
                    "POST",
                    "/v1/completions",
                    json=request.dict(),
                    # Per-request timeout override is possible here too:
                    # timeout=Timeout(10.0, read=120.0)
                ) as response:
                    response.raise_for_status()  # Check for HTTP errors (4xx, 5xx)
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
                # Re-raise or handle as appropriate for your application
                raise
            except httpx.HTTPStatusError as e:
                # Log details including response body if possible and safe
                error_body = "N/A"
                try:
                    # Read response body asynchronously ONLY if needed and safe
                    # (beware of large bodies)
                    # error_body = await e.response.aread()
                    # For logging, maybe just read the first few hundred bytes?
                    # Or rely on the default repr which includes status code and url
                    pass
                except Exception:
                    pass  # Ignore errors reading the body
                logging_utility.error(
                    "HTTP error during streaming completions: %s - Status: %s, Response: %s",
                    str(e),
                    e.response.status_code,
                    error_body,
                )
                raise
            except httpx.RequestError as e:
                # Handles other request errors like connection issues
                logging_utility.error(
                    "Network or request error during streaming completions: %s", str(e)
                )
                raise
            except Exception as e:
                logging_utility.error(
                    "Unexpected error during streaming completions: %s",
                    str(e),
                    exc_info=True,  # Add traceback info for unexpected errors
                )
                raise

    def close(self):
        """Closes the underlying synchronous HTTPX client."""
        if self.client and not self.client.is_closed:
            self.client.close()
            logging_utility.info("InferenceClient synchronous client closed.")
