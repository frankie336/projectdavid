import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

ent_validator = ValidationInterface()


load_dotenv()

logging_utility = UtilsInterface.LoggingUtility()


class MessagesClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the MessagesClient with the given base URL and optional API key.
        Uses X-API-Key authentication for consistency across all internal clients.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            logging_utility.info("API Key provided and added to headers.")
        else:
            logging_utility.warning(
                "No API Key provided. Access to protected endpoints may be denied."
            )

        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
        )

        self.message_chunks: Dict[str, List[str]] = {}
        logging_utility.info(
            "MessagesClient initialized with base_url: %s", self.base_url
        )

    def create_message(
        self,
        thread_id: str,
        content: str,
        assistant_id: str,
        role: str = "user",
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> ent_validator.MessageRead:
        """
        Create a new message and return it as a MessageRead model.

        Args:
            thread_id (str): ID of the thread.
            content (str): Message content.
            assistant_id (str): Assistant's ID.
            role (str): Message role, default 'user'.
            meta_data (Optional[Dict[str, Any]]): Additional metadata.

        Returns:
            MessageRead: The created message.
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
            "Creating message for thread_id: %s, role: %s", thread_id, role
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
        """
        Retrieve a message by its ID.

        Args:
            message_id (str): The ID of the message.

        Returns:
            MessageRead: The retrieved message.
        """
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
        """
        Update an existing message with provided fields.

        Args:
            message_id (str): The ID of the message.
            **updates: Fields to update.

        Returns:
            MessageRead: The updated message.
        """
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
        self, thread_id: str, limit: int = 20, order: str = "asc"
    ) -> List[Dict[str, Any]]:
        """
        List messages for a given thread.

        Args:
            thread_id (str): The thread ID.
            limit (int): Maximum number of messages to retrieve.
            order (str): Order of messages ('asc' or 'desc').

        Returns:
            List[Dict[str, Any]]: A list of messages as dictionaries.
        """
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
            messages = response.json()
            validated_messages = [
                ent_validator.MessageRead(**message) for message in messages
            ]
            logging_utility.info("Retrieved %d messages", len(validated_messages))
            return [message.dict() for message in validated_messages]
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while listing messages: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while listing messages: %s", str(e)
            )
            raise

    def get_formatted_messages(
        self, thread_id: str, system_message: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and format messages for a thread, inserting or replacing the system message.

        Args:
            thread_id (str): The thread ID.
            system_message (str): The system message to use.

        Returns:
            List[Dict[str, Any]]: The formatted list of messages.
        """
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
        """
        Retrieve formatted messages for a thread without modifying the system message.

        Args:
            thread_id (str): The thread ID.

        Returns:
            List[Dict[str, Any]]: The list of formatted messages.
        """
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

    def delete_message(self, message_id: str) -> Dict[str, Any]:
        """
        Delete a message by its ID.

        Args:
            message_id (str): The ID of the message.

        Returns:
            Dict[str, Any]: The deletion result.
        """
        logging_utility.info("Deleting message with id: %s", message_id)
        try:
            response = self.client.delete(f"/v1/messages/{message_id}")
            response.raise_for_status()
            result = response.json()
            logging_utility.info("Message deleted successfully")
            return result
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while deleting message: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while deleting message: %s", str(e)
            )
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
        """
        Save a message chunk from the assistant, with support for streaming and dynamic roles.

        Args:
            thread_id (str): The thread ID.
            role (str): The role (e.g., 'assistant', 'user', 'system').
            content (str): The message content.
            assistant_id (str): The assistant's ID.
            sender_id (str): The ID of the sender.
            is_last_chunk (bool): Whether this is the final chunk.
            meta_data (Optional[Dict[str, Any]]): Additional metadata.

        Returns:
            Optional[MessageRead]: The final saved message for final chunks, None otherwise.
        """
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
        role: str = "tool",
        sender_id: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit tool output as a message.

        Args:
            thread_id (str): The thread ID.
            content (str): The message content.
            assistant_id (str): The assistant's ID.
            tool_id (str): The tool's ID.
            role (str): The role, default 'tool'.
            sender_id (Optional[str]): Optional sender ID.
            meta_data (Optional[Dict[str, Any]]): Additional metadata.

        Returns:
            Dict[str, Any]: The created message data.
        """
        meta_data = meta_data or {}
        message_data = {
            "thread_id": thread_id,
            "content": content,
            "role": role,
            "assistant_id": assistant_id,
            "tool_id": tool_id,
            "meta_data": meta_data,
        }
        if sender_id is not None:
            message_data["sender_id"] = sender_id

        logging_utility.info(
            "Creating tool message for thread_id: %s, role: %s", thread_id, role
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
