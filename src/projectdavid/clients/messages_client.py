# projectdavid/clients/messages_client.py
import base64
from typing import Any, Dict, List, Optional, Union

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

ent_validator = ValidationInterface()
from projectdavid.clients.base_client import BaseAPIClient

load_dotenv()

logging_utility = UtilsInterface.LoggingUtility()

# Mirrors ContentType in messages_schema.py
ContentType = Union[str, List[Dict[str, Any]]]


class MessagesClient(BaseAPIClient):
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(base_url=base_url, api_key=api_key)
        self.message_chunks: Dict[str, List[str]] = {}
        logging_utility.info("MessagesClient initialized using BaseAPIClient.")

    # ──────────────────────────────────────────────────────────────────────────
    #  Multimodal builder
    #
    #  Developers NEVER construct content blocks by hand.
    #  They call build_image_message() and pass the result to create_message().
    #
    #  Usage:
    #
    #      # Text only — unchanged from before
    #      entity.messages.create_message(
    #          thread_id=thread_id,
    #          content="What is the weather?",
    #          assistant_id=assistant_id,
    #      )
    #
    #      # Multimodal — image from disk
    #      content = entity.messages.build_image_message(
    #          text="What is in this image?",
    #          image_bytes=open("photo.jpg", "rb").read(),
    #          mime_type="image/jpeg",
    #      )
    #      entity.messages.create_message(
    #          thread_id=thread_id,
    #          content=content,
    #          assistant_id=assistant_id,
    #      )
    #
    #      # Multimodal — image already in memory (e.g. from requests / httpx)
    #      content = entity.messages.build_image_message(
    #          text="Describe this chart.",
    #          image_bytes=response.content,
    #          mime_type="image/png",
    #      )
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_image_message(
        text: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
    ) -> List[Dict[str, Any]]:
        """
        Build a multimodal content list from raw image bytes.

        Args:
            text:        The text prompt to accompany the image.
            image_bytes: Raw bytes of the image (JPEG, PNG, WEBP, GIF).
            mime_type:   MIME type string, default "image/jpeg".
                         Supported: "image/jpeg", "image/png",
                                    "image/webp", "image/gif"

        Returns:
            A content list ready to pass directly to create_message(content=...).

        Example:
            content = MessagesClient.build_image_message(
                text="What breed is this dog?",
                image_bytes=open("dog.jpg", "rb").read(),
                mime_type="image/jpeg",
            )
        """
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{encoded}"
        return [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]

    # ──────────────────────────────────────────────────────────────────────────
    #  Core message operations
    # ──────────────────────────────────────────────────────────────────────────

    def create_message(
        self,
        thread_id: str,
        content: ContentType,  # ← str for text, or list from build_image_message()
        assistant_id: str,
        role: str = "user",
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> ent_validator.MessageRead:
        """
        Create a new message.

        For plain text pass content as a string (unchanged from before).
        For vision messages pass the list returned by build_image_message().
        """
        meta_data = meta_data or {}
        message_data = {
            "thread_id": thread_id,
            "content": content,
            "role": role,
            "assistant_id": assistant_id,
            "meta_data": meta_data,
        }

        logging_utility.info(
            "Creating message for thread_id: %s, role: %s, content_type: %s",
            thread_id,
            role,
            "multimodal" if isinstance(content, list) else "text",
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
            return ent_validator.MessageRead(**response.json())
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
            logging_utility.info("Message updated successfully")
            return ent_validator.MessageRead(**response.json())
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
        self, thread_id: str, limit: int = 20, order: str = "asc"
    ) -> ent_validator.MessagesList:
        logging_utility.info(
            "Listing messages for thread_id: %s, limit: %d, order: %s",
            thread_id,
            limit,
            order,
        )
        try:
            response = self.client.get(
                f"/v1/threads/{thread_id}/messages",
                params={"limit": limit, "order": order},
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
        try:
            response = self.client.get(f"/v1/threads/{thread_id}/formatted_messages")
            response.raise_for_status()
            formatted_messages = response.json()
            if not isinstance(formatted_messages, list):
                raise ValueError("Expected a list of messages")
            for msg in formatted_messages:
                if msg.get("role") == "tool":
                    if "tool_call_id" not in msg or "content" not in msg:
                        logging_utility.warning(
                            "Malformed tool message detected: %s", msg
                        )
                        raise ValueError(f"Malformed tool message: {msg}")
            if formatted_messages and formatted_messages[0].get("role") == "system":
                formatted_messages[0]["content"] = system_message
            else:
                formatted_messages.insert(
                    0, {"role": "system", "content": system_message}
                )
            logging_utility.info(
                "Retrieved %d formatted messages", len(formatted_messages)
            )
            return formatted_messages
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Thread not found: {thread_id}")
            raise RuntimeError(f"HTTP error occurred: {e}")
        except Exception as e:
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
            return formatted_messages
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Thread not found: {thread_id}")
            raise RuntimeError(f"HTTP error occurred: {e}")
        except Exception as e:
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
