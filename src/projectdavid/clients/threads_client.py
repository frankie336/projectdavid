# projectdavid/clients/threads_client.py

import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

from projectdavid.clients.base_client import BaseAPIClient

load_dotenv()
validator = ValidationInterface()
logging_utility = UtilsInterface.LoggingUtility()


class ThreadsClient(BaseAPIClient):
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(base_url=base_url, api_key=api_key)
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
            return validator.UserRead(**created_user)
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error occurred while creating user: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("Unexpected error creating user: %s", str(e))
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
            return validator.ThreadRead(**created_thread)
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
            logging_utility.error("Unexpected error creating thread: %s", str(e))
            raise

    def retrieve_thread(self, thread_id: str) -> validator.ThreadRead:
        logging_utility.info("Retrieving thread with id: %s", thread_id)
        try:
            response = self.client.get(f"/v1/threads/{thread_id}")
            response.raise_for_status()
            thread = response.json()
            return validator.ThreadRead(**thread)
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error retrieving thread: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("Unexpected error retrieving thread: %s", str(e))
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
            logging_utility.error("HTTP error updating thread: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("Unexpected error updating thread: %s", str(e))
            raise

    def update_thread_metadata(
        self, thread_id: str, new_metadata: Dict[str, Any]
    ) -> validator.ThreadRead:
        try:
            thread = self.retrieve_thread(thread_id)
            current_metadata = thread.meta_data
            current_metadata.update(new_metadata)
            return self.update_thread(thread_id, meta_data=current_metadata)
        except Exception as e:
            logging_utility.error("Error updating thread metadata: %s", str(e))
            raise

    def list_threads(self, user_id: str) -> List[str]:
        logging_utility.info("Listing threads for user: %s", user_id)
        try:
            response = self.client.get(f"/v1/users/{user_id}/threads")
            response.raise_for_status()
            thread_ids = response.json()
            return validator.ThreadIds(**thread_ids).thread_ids
        except Exception as e:
            logging_utility.error("Error listing threads: %s", str(e))
            raise

    def delete_thread(self, thread_id: str) -> bool:
        logging_utility.info("Deleting thread with id: %s", thread_id)
        try:
            response = self.client.delete(f"/v1/threads/{thread_id}")
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            logging_utility.error("HTTP error deleting thread: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("Unexpected error deleting thread: %s", str(e))
            raise
