import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

from projectdavid.clients.base_client import BaseAPIClient

ent_validator = ValidationInterface()

load_dotenv()

logging_utility = UtilsInterface.LoggingUtility()


class MessagesClient(BaseAPIClient):
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        MessagesClient for interacting with the /v1/messages API.

        Inherits BaseAPIClient for consistent header injection,
        timeout configuration, and error logging.
        """
        super().__init__(base_url=base_url, api_key=api_key)
        self.message_chunks: Dict[str, List[str]] = {}

        # Lazy-loaded FileClient so we only instantiate it if multimodal content is passed
        self._file_client = None

        logging_utility.info("MessagesClient initialized using BaseAPIClient.")

    @property
    def file_client(self):
        """Lazy load the FileClient to prevent circular imports on startup."""
        if self._file_client is None:
            # Import FileClient locally here assuming it's in the same package
            from projectdavid.clients.files_client import FileClient

            self._file_client = FileClient(base_url=self.base_url, api_key=self.api_key)
        return self._file_client

    def _process_and_upload_image(self, url: str, index: int) -> Dict[str, Any]:
        """
        Helper to download/decode a single image and upload it via FileClient.
        """
        file_bytes = b""
        filename = f"upload_image_{index}.jpg"

        try:
            if url.startswith("data:image"):
                # 1. Handle Base64
                header, encoded = url.split(",", 1)
                file_bytes = base64.b64decode(encoded)
                if "image/png" in header:
                    filename = f"upload_image_{index}.png"
                elif "image/webp" in header:
                    filename = f"upload_image_{index}.webp"
            elif url.startswith("http"):
                # 2. Handle standard URLs
                # Add a standard User-Agent so strict servers (like Wikipedia) don't block us with a 403
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                }
                response = httpx.get(url, headers=headers, timeout=30.0)
                response.raise_for_status()
                file_bytes = response.content
            else:
                raise ValueError(f"Unsupported image URL format at index {index}")

            # 3. Upload to File Server via FileClient
            file_obj = io.BytesIO(file_bytes)
            file_response = self.file_client.upload_file_object(
                file_object=file_obj, file_name=filename, purpose="vision"
            )

            # 4. Return the standard attachment format
            return {"type": "image", "file_id": file_response.id}

        except Exception as e:
            logging_utility.error("Failed to process image %d: %s", index, str(e))
            raise RuntimeError(f"Failed to process image {index}: {str(e)}")

    def prepare_multimodal_payload(
        self, content: Union[str, List[Dict[str, Any]]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Pre-parses the array, batch-uploads images synchronously via threads,
        and returns a DB-safe text string & attachments list.
        """
        # If it's already a plain string, no work needed.
        if isinstance(content, str):
            return content, []

        text_blocks = []
        image_tasks = []
        attachments = []

        # 1. Parse the array and separate Text vs Images
        for i, block in enumerate(content):
            block_type = block.get("type")

            if block_type in ("text", "input_text"):
                text_blocks.append(block.get("text", ""))

            elif block_type in ("image_url", "input_image"):
                img_data = block.get("image_url")
                url = img_data.get("url") if isinstance(img_data, dict) else img_data

                # Queue the upload task for concurrent batching
                image_tasks.append((url, i))

            elif block_type == "image_file":
                # If the user already uploaded it and passed a file_id directly
                file_id = block.get("image_file", {}).get("file_id")
                if file_id:
                    attachments.append({"type": "image", "file_id": file_id})

        # 2. Execute all uploads concurrently using ThreadPoolExecutor
        if image_tasks:
            logging_utility.info(
                "Batch uploading %d images for multimodal payload...", len(image_tasks)
            )
            with ThreadPoolExecutor(max_workers=min(len(image_tasks), 10)) as executor:
                # Map futures to tasks
                futures = [
                    executor.submit(self._process_and_upload_image, task[0], task[1])
                    for task in image_tasks
                ]

                for future in as_completed(futures):
                    try:
                        uploaded_attachment = future.result()
                        attachments.append(uploaded_attachment)
                    except Exception as e:
                        logging_utility.error("Batch image upload failed: %s", str(e))
                        raise

        # 3. Compile the text blocks into a single string for your DB
        final_text = "\n\n".join(text_blocks).strip()

        return final_text, attachments

    def create_message(
        self,
        thread_id: str,
        content: Union[str, List[Dict[str, Any]]],  # <--- UPDATED TYPE
        assistant_id: str,
        role: str = "user",
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> ent_validator.MessageRead:
        """
        Create a new message and return it as a MessageRead model.
        Automatically intercepts multimodal arrays, uploads files, and cleanly formats DB payload.
        """
        # 1. Clean up SDK side: Download, upload, and extract
        clean_content, attachments = self.prepare_multimodal_payload(content)

        meta_data = meta_data or {}
        message_data = {
            "thread_id": thread_id,
            "content": clean_content,
            "role": role,
            "assistant_id": assistant_id,
            "meta_data": meta_data,
            "attachments": attachments,  # <--- INJECTED ATTACHMENTS
        }

        logging_utility.info(
            "Creating message for thread_id: %s, role: %s, attachments: %d",
            thread_id,
            role,
            len(attachments),
        )

        try:
            validated_data = ent_validator.MessageCreate(**message_data)
            response = self.client.post("/v1/messages", json=validated_data.dict())
            response.raise_for_status()

            created_message = response.json()
            logging_utility.info(
                "Message created successfully with id: %s", created_message.get("id")
            )
            return ent_validator.MessageRead(**created_message)

        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while creating message: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while creating message: %s", str(e)
            )
            raise

    def retrieve_message(self, message_id: str) -> ent_validator.MessageRead:
        logging_utility.info("Retrieving message with id: %s", message_id)
        try:
            response = self.client.get(f"/v1/messages/{message_id}")
            response.raise_for_status()
            message = response.json()
            logging_utility.info("Message retrieved successfully")
            return ent_validator.MessageRead(**message)
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while retrieving message: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while retrieving message: %s", str(e)
            )
            raise

    def update_message(self, message_id: str, **updates) -> ent_validator.MessageRead:
        logging_utility.info("Updating message with id: %s", message_id)
        try:
            validated_data = ent_validator.MessageUpdate(**updates)
            response = self.client.put(
                f"/v1/messages/{message_id}",
                json=validated_data.dict(exclude_unset=True),
            )
            response.raise_for_status()
            updated_message = response.json()
            logging_utility.info("Message updated successfully")
            return ent_validator.MessageRead(**updated_message)
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while updating message: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while updating message: %s", str(e)
            )
            raise

    def list_messages(
        self,
        thread_id: str,
        limit: int = 20,
        order: str = "asc",
    ) -> ent_validator.MessagesList:
        logging_utility.info(
            "Listing messages for thread_id: %s, limit: %d, order: %s",
            thread_id,
            limit,
            order,
        )
        params = {"limit": limit, "order": order}
        try:
            response = self.client.get(
                f"/v1/threads/{thread_id}/messages", params=params
            )
            response.raise_for_status()

            envelope = ent_validator.MessagesList(**response.json())
            logging_utility.info("Retrieved %d messages", len(envelope.data))
            return envelope

        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}") from e
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error while listing messages: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("Unexpected error while listing messages: %s", str(e))
            raise

    def get_formatted_messages(
        self, thread_id: str, system_message: str = ""
    ) -> List[Dict[str, Any]]:
        logging_utility.info("Getting formatted messages for thread_id: %s", thread_id)
        logging_utility.info("Using system message: %s", system_message)
        try:
            response = self.client.get(f"/v1/threads/{thread_id}/formatted_messages")
            response.raise_for_status()
            formatted_messages = response.json()
            if not isinstance(formatted_messages, list):
                raise ValueError("Expected a list of messages")
            logging_utility.debug("Initial formatted messages: %s", formatted_messages)
            for msg in formatted_messages:
                if msg.get("role") == "tool":
                    if "tool_call_id" not in msg or "content" not in msg:
                        logging_utility.warning(
                            "Malformed tool message detected: %s", msg
                        )
                        raise ValueError(f"Malformed tool message: {msg}")
            if formatted_messages and formatted_messages[0].get("role") == "system":
                formatted_messages[0]["content"] = system_message
                logging_utility.debug(
                    "Replaced existing system message with: %s", system_message
                )
            else:
                formatted_messages.insert(
                    0, {"role": "system", "content": system_message}
                )
                logging_utility.debug("Inserted new system message: %s", system_message)
            logging_utility.info(
                "Formatted messages after insertion: %s", formatted_messages
            )
            logging_utility.info(
                "Retrieved %d formatted messages", len(formatted_messages)
            )
            return formatted_messages
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logging_utility.error("Thread not found: %s", thread_id)
                raise ValueError(f"Thread not found: {thread_id}")
            else:
                logging_utility.error("HTTP error occurred: %s", str(e))
                raise RuntimeError(f"HTTP error occurred: {e}")
        except Exception as e:
            logging_utility.error("An error occurred: %s", str(e))
            raise RuntimeError(f"An error occurred: {str(e)}")

    def get_messages_without_system_message(
        self, thread_id: str
    ) -> List[Dict[str, Any]]:
        logging_utility.info(
            "Getting messages without system message for thread_id: %s", thread_id
        )
        try:
            response = self.client.get(f"/v1/threads/{thread_id}/formatted_messages")
            response.raise_for_status()
            formatted_messages = response.json()
            if not isinstance(formatted_messages, list):
                raise ValueError("Expected a list of messages")
            logging_utility.debug(
                "Retrieved formatted messages: %s", formatted_messages
            )
            logging_utility.info(
                "Retrieved %d formatted messages", len(formatted_messages)
            )
            return formatted_messages
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logging_utility.error("Thread not found: %s", thread_id)
                raise ValueError(f"Thread not found: {thread_id}")
            else:
                logging_utility.error("HTTP error occurred: %s", str(e))
                raise RuntimeError(f"HTTP error occurred: {e}")
        except Exception as e:
            logging_utility.error("An error occurred: %s", str(e))
            raise RuntimeError(f"An error occurred: {str(e)}")

    def delete_message(self, message_id: str) -> ent_validator.MessageDeleted:
        logging_utility.info("Deleting message with id: %s", message_id)
        try:
            response = self.client.delete(f"/v1/messages/{message_id}")
            response.raise_for_status()
            return ent_validator.MessageDeleted(**response.json())
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error while deleting message: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("Unexpected error while deleting message: %s", str(e))
            raise

    def save_assistant_message_chunk(
        self,
        thread_id: str,
        role: str,
        content: str,
        assistant_id: str,
        sender_id: str,
        is_last_chunk: bool = False,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[ent_validator.MessageRead]:
        logging_utility.info(
            "Saving assistant message chunk for thread_id: %s, role: %s, is_last_chunk: %s",
            thread_id,
            role,
            is_last_chunk,
        )
        message_data = {
            "thread_id": thread_id,
            "content": content,
            "role": role,
            "assistant_id": assistant_id,
            "sender_id": sender_id,
            "is_last_chunk": is_last_chunk,
            "meta_data": meta_data or {},
        }
        try:
            response = self.client.post("/v1/messages/assistant", json=message_data)
            response.raise_for_status()
            if is_last_chunk:
                message_read = ent_validator.MessageRead(**response.json())
                logging_utility.info(
                    "Final assistant message chunk saved successfully. Message ID: %s",
                    message_read.id,
                )
                return message_read
            else:
                logging_utility.info(
                    "Non-final assistant message chunk saved successfully."
                )
                return None
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error while saving assistant message chunk: %s (Status: %d)",
                str(e),
                e.response.status_code,
            )
            return None
        except Exception as e:
            logging_utility.error(
                "Unexpected error while saving assistant message chunk: %s", str(e)
            )
            return None

    def submit_tool_output(
        self,
        thread_id: str,
        content: str,
        assistant_id: str,
        tool_id: str,
        tool_call_id: Optional[str] = None,
        role: str = "tool",
        sender_id: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta_data = meta_data or {}
        message_data = {
            "thread_id": thread_id,
            "content": content,
            "role": role,
            "assistant_id": assistant_id,
            "tool_id": tool_id,
            "meta_data": meta_data,
        }

        if tool_call_id:
            message_data["tool_call_id"] = tool_call_id

        if sender_id is not None:
            message_data["sender_id"] = sender_id

        logging_utility.info(
            "Creating tool message for thread_id: %s, role: %s, call_id: %s",
            thread_id,
            role,
            tool_call_id,
        )

        try:
            validated_data = ent_validator.MessageCreate(**message_data)
            response = self.client.post(
                "/v1/messages/tools", json=validated_data.dict()
            )
            response.raise_for_status()
            created_message = response.json()

            logging_utility.info(
                "Tool message created successfully with id: %s",
                created_message.get("id"),
            )
            return created_message

        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while creating tool message: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while creating tool message: %s", str(e)
            )
            raise
