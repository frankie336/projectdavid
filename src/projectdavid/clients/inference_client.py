import asyncio
import json
import os
import time
from typing import AsyncGenerator, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

ent_validator = ValidationInterface()
load_dotenv()
logging_utility = UtilsInterface.LoggingUtility()


class InferenceClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the InferenceClient with a base URL and an optional API key.
        Uses X-API-Key authentication header, consistent with other clients.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY")

        self.timeout = httpx.Timeout(timeout=60.0, connect=10.0, read=30.0, write=30.0)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            logging_utility.info("API Key provided and added to headers.")
        else:
            logging_utility.warning(
                "No API Key provided. Access to protected endpoints may be denied."
            )

        self.client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=self.timeout
        )

        logging_utility.info(
            "InferenceClient initialized with base_url: %s", self.base_url
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
        """
        Synchronously aggregates the streaming completions result and returns the JSON completion.
        Internally, it uses the asynchronous stream_inference_response method in a background thread.
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
            validated_payload = ent_validator.StreamRequest(**payload)
        except ValidationError as e:
            logging_utility.error("Payload validation error: %s", e.json())
            raise ValueError(f"Payload validation error: {e}")

        logging_utility.info(
            "Sending completions request (sync wrapper): %s", validated_payload.dict()
        )

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

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            final_content = loop.run_until_complete(aggregate())
        finally:
            loop.close()

        completions_response = {
            "id": f"chatcmpl-{run_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
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
                "prompt_tokens": 0,
                "completion_tokens": len(final_content.split()),
                "total_tokens": len(final_content.split()),
            },
        }
        return completions_response

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
        Initiates an asynchronous streaming request to the completions
        endpoint and yields each response chunk as a dict.
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
            validated_payload = ent_validator.StreamRequest(**payload)
        except ValidationError as e:
            logging_utility.error("Payload validation error: %s", e.json())
            raise ValueError(f"Payload validation error: {e}")

        logging_utility.info(
            "Sending streaming inference request: %s", validated_payload.dict()
        )

        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout
        ) as async_client:
            if self.api_key:
                async_client.headers["Authorization"] = f"Bearer {self.api_key}"

            try:
                async with async_client.stream(
                    "POST", "/v1/completions", json=validated_payload.dict()
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
                                    "Error decoding JSON from stream: %s", str(json_exc)
                                )
                                continue
            except httpx.HTTPStatusError as e:
                logging_utility.error(
                    "HTTP error during streaming completions: %s", str(e)
                )
                raise
            except Exception as e:
                logging_utility.error(
                    "Unexpected error during streaming completions: %s", str(e)
                )
                raise

    def close(self):
        """
        Closes the underlying synchronous HTTP client.
        """
        self.client.close()
