import os
import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.constants.timeouts import DEFAULT_TIMEOUT
from pydantic import ValidationError

from projectdavid.clients.base_client import BaseAPIClient

ent_validator = ValidationInterface()


# Load environment variables
load_dotenv()

logging_utility = UtilsInterface.LoggingUtility()


class AssistantsClientError(Exception):
    """Custom exception for AssistantsClient errors."""

    pass


class AssistantsClient(BaseAPIClient):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        write_timeout: float = 30.0,
    ):
        """
        AssistantsClient constructor using BaseAPIClient configuration.
        Inherits base_url, api_key, timeouts, headers, and client instantiation.
        """
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )
        logging_utility.info("AssistantsClient ready at: %s", self.base_url)

    def close(self):
        """Closes the HTTP client session."""
        self.client.close()

    @staticmethod
    def _parse_response(response):
        """Parses JSON responses safely."""
        try:
            return response.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error("API returned HTTP error: %s", str(e))
            raise
        except httpx.DecodingError:
            logging_utility.error("Failed to decode JSON response: %s", response.text)
            raise AssistantsClientError("Invalid JSON response from API.")

    def _request_with_retries(self, method: str, url: str, **kwargs):
        """Handles retries for transient failures."""
        global response
        retries = 3
        for attempt in range(retries):
            try:
                response = self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError:
                if response.status_code in {500, 503} and attempt < retries - 1:
                    logging_utility.warning(
                        "Retrying request due to server error (attempt %d)", attempt + 1
                    )
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    raise

    def create_assistant(
        self,
        model: str = "",
        name: str = "",
        description: str = "",
        instructions: str = "",
        meta_data: Dict[str, Any] = None,
        top_p: float = 1.0,
        temperature: float = 1.0,
        response_format: str = "auto",
        assistant_id: Optional[str] = None,
    ) -> ent_validator.AssistantRead:
        """Creates an assistant.
        :type response_format: object
        """
        assistant_data = {
            "id": assistant_id,
            "name": name,
            "description": description,
            "model": model,
            "instructions": instructions,
            "meta_data": meta_data,
            "top_p": top_p,
            "temperature": temperature,
            "response_format": response_format,
        }

        try:
            validated_data = ent_validator.AssistantCreate(**assistant_data)
            logging_utility.info(
                "Creating assistant with model: %s, name: %s", model, name
            )

            response = self._request_with_retries(
                "POST", "/v1/assistants", json=validated_data.model_dump()
            )
            created_assistant = self._parse_response(response)

            validated_response = ent_validator.AssistantRead(**created_assistant)
            logging_utility.info(
                "Assistant created successfully with id: %s", validated_response.id
            )
            return validated_response
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise AssistantsClientError(f"Validation error: {e}")

    def retrieve_assistant(self, assistant_id: str) -> ent_validator.AssistantRead:
        """Retrieves an assistant by ID."""
        logging_utility.info("Retrieving assistant with id: %s", assistant_id)
        try:
            response = self._request_with_retries(
                "GET", f"/v1/assistants/{assistant_id}"
            )
            assistant = self._parse_response(response)

            validated_data = ent_validator.AssistantRead(**assistant)
            logging_utility.info("Assistant retrieved successfully")
            return validated_data
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise AssistantsClientError(f"Validation error: {e}")

    def update_assistant(
        self, assistant_id: str, **updates
    ) -> ent_validator.AssistantRead:
        """Updates an assistant."""
        logging_utility.info("Updating assistant with id: %s", assistant_id)
        try:
            updates.pop("id", None)
            updates.pop("assistant_id", None)

            validated_data = ent_validator.AssistantUpdate(**updates)

            response = self._request_with_retries(
                "PUT",
                f"/v1/assistants/{assistant_id}",
                json=validated_data.model_dump(exclude_unset=True),
            )
            updated_assistant = self._parse_response(response)

            validated_response = ent_validator.AssistantRead(**updated_assistant)
            logging_utility.info("Assistant updated successfully")
            return validated_response
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise AssistantsClientError(f"Validation error: {e}")

    def delete_assistant(self, assistant_id: str) -> Dict[str, Any]:
        """Deletes an assistant by ID."""
        logging_utility.info("Deleting assistant with id: %s", assistant_id)
        response = self._request_with_retries(
            "DELETE", f"/v1/assistants/{assistant_id}"
        )
        return self._parse_response(response)

    def associate_assistant_with_user(
        self, user_id: str, assistant_id: str
    ) -> Dict[str, Any]:
        """Associates an assistant with a user."""
        logging_utility.info(
            "Associating assistant %s with user %s", assistant_id, user_id
        )
        self._request_with_retries(
            "POST", f"/v1/users/{user_id}/assistants/{assistant_id}"
        )
        return {"message": "Assistant associated with user successfully"}

    def disassociate_assistant_from_user(
        self, user_id: str, assistant_id: str
    ) -> Dict[str, Any]:
        """Disassociates an assistant from a user."""
        logging_utility.info(
            "Disassociating assistant %s from user %s", assistant_id, user_id
        )
        self._request_with_retries(
            "DELETE", f"/v1/users/{user_id}/assistants/{assistant_id}"
        )
        return {"message": "Assistant disassociated from user successfully"}

    def list_assistants_by_user(
        self, user_id: str
    ) -> List[ent_validator.AssistantRead]:
        """Lists all assistants associated with a user."""
        logging_utility.info("Retrieving assistants for user id: %s", user_id)
        try:
            response = self._request_with_retries(
                "GET", f"/v1/users/{user_id}/assistants"
            )
            assistants = self._parse_response(response)

            validated_assistants = [
                ent_validator.AssistantRead(**assistant) for assistant in assistants
            ]
            logging_utility.info(
                "Assistants retrieved successfully for user id: %s", user_id
            )
            return validated_assistants
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise AssistantsClientError(f"Validation error: {e}")
