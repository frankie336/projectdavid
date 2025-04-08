from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

validator = ValidationInterface()

load_dotenv()
logging_utility = UtilsInterface.LoggingUtility()


class ThreadsClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the ThreadsClient with the given base URL and optional API key.

        Args:
            base_url (str): The base URL for the threads service.
            api_key (Optional[str]): The API key for authentication.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        logging_utility.info(
            "ThreadsClient initialized with base_url: %s", self.base_url
        )

    def create_user(self, name: str) -> validator.UserRead:
        logging_utility.info("Creating user with name: %s", name)
        user_data = validator.UserCreate(name=name).model_dump()
        try:
            response = self.client.post("/v1/users", json=user_data)
            response.raise_for_status()
            created_user = response.json()
            validated_user = validator.UserRead(**created_user)
            logging_utility.info(
                "User created successfully with id: %s", validated_user.id
            )
            return validated_user
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error occurred while creating user: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("An error occurred while creating user: %s", str(e))
            raise

    def create_thread(
        self, participant_ids: List[str], meta_data: Optional[Dict[str, Any]] = None
    ) -> validator.ThreadRead:
        meta_data = meta_data or {}
        thread_data = validator.ThreadCreate(
            participant_ids=participant_ids, meta_data=meta_data
        ).model_dump()
        logging_utility.info(
            "Creating thread with %d participants", len(participant_ids)
        )
        try:
            response = self.client.post("/v1/threads", json=thread_data)
            response.raise_for_status()
            created_thread = response.json()
            validated_thread = validator.ThreadRead(**created_thread)
            logging_utility.info(
                "Thread created successfully with id: %s", validated_thread.id
            )
            return validated_thread
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while creating thread: %s", str(e)
            )
            logging_utility.error(
                "Status code: %d, Response text: %s",
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error("An error occurred while creating thread: %s", str(e))
            raise

    def retrieve_thread(self, thread_id: str) -> validator.ThreadRead:
        logging_utility.info("Retrieving thread with id: %s", thread_id)
        try:
            response = self.client.get(f"/v1/threads/{thread_id}")
            response.raise_for_status()
            thread = response.json()
            validated_thread = validator.ThreadRead(**thread)
            logging_utility.info("Thread retrieved successfully")
            return validated_thread
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while retrieving thread: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while retrieving thread: %s", str(e)
            )
            raise

    def update_thread(self, thread_id: str, **updates) -> validator.ThreadReadDetailed:
        logging_utility.info("Updating thread with id: %s", thread_id)
        try:
            validated_updates = validator.ThreadUpdate(**updates)
            response = self.client.post(
                f"/v1/threads/{thread_id}", json=validated_updates.model_dump()
            )
            response.raise_for_status()
            updated_thread = response.json()
            return validator.ThreadReadDetailed(**updated_thread)
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while updating thread: %s", str(e)
            )
            logging_utility.error("Response content: %s", e.response.content)
            raise
        except Exception as e:
            logging_utility.error("An error occurred while updating thread: %s", str(e))
            raise

    def update_thread_metadata(
        self, thread_id: str, new_metadata: Dict[str, Any]
    ) -> validator.ThreadRead:
        logging_utility.info("Updating metadata for thread with id: %s", thread_id)
        try:
            thread = self.retrieve_thread(thread_id)
            current_metadata = thread.meta_data
            current_metadata.update(new_metadata)
            return self.update_thread(thread_id, meta_data=current_metadata)
        except ValidationError as e:
            logging_utility.error(
                "Validation error while updating thread metadata: %s", e.json()
            )
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while updating thread metadata: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while updating thread metadata: %s", str(e)
            )
            raise

    def list_threads(self, user_id: str) -> List[str]:
        logging_utility.info("Listing threads for user with id: %s", user_id)
        try:
            response = self.client.get(f"/v1/users/{user_id}/threads")
            response.raise_for_status()
            thread_ids = response.json()
            validated_thread_ids = validator.ThreadIds(**thread_ids)
            logging_utility.info(
                "Retrieved %d thread ids", len(validated_thread_ids.thread_ids)
            )
            return validated_thread_ids.thread_ids
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while listing threads: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error("An error occurred while listing threads: %s", str(e))
            raise

    def delete_thread(self, thread_id: str) -> bool:
        logging_utility.info("Deleting thread with id: %s", thread_id)
        try:
            response = self.client.delete(f"/v1/threads/{thread_id}")
            response.raise_for_status()
            logging_utility.info("Thread deleted successfully")
            return True
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while deleting thread: %s", str(e)
            )
            if e.response.status_code == 404:
                return False
            raise
