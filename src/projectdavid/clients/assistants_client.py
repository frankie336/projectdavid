import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.constants.timeouts import DEFAULT_TIMEOUT  # noqa: F401
from pydantic import ValidationError

from projectdavid.clients.base_client import BaseAPIClient

ent_validator = ValidationInterface()

# Load environment variables
load_dotenv()
logging_utility = UtilsInterface.LoggingUtility()


class AssistantsClientError(Exception):
    """Custom exception for AssistantsClient errors."""


class AssistantsClient(BaseAPIClient):
    # ------------------------------------------------------------------ #
    #  INIT / SESSION
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        write_timeout: float = 30.0,
    ):
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
        self.client.close()

    # ------------------------------------------------------------------ #
    #  INTERNAL HELPERS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(response: httpx.Response):
        try:
            return response.json()
        except httpx.DecodingError:
            logging_utility.error("Failed to decode JSON response: %s", response.text)
            raise AssistantsClientError("Invalid JSON response from API.")

    def _request_with_retries(self, method: str, url: str, **kwargs) -> httpx.Response:
        retries = 3
        for attempt in range(retries):
            try:
                resp = self.client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {500, 503} and attempt < retries - 1:
                    logging_utility.warning(
                        "Retrying request due to server error (attempt %d)", attempt + 1
                    )
                    time.sleep(2**attempt)
                else:
                    raise

    # ------------------------------------------------------------------ #
    #  CRUD
    # ------------------------------------------------------------------ #
    def create_assistant(
        self,
        *,
        model: str = "",
        name: str = "",
        description: str = "",
        instructions: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        platform_tools: Optional[List[Dict[str, Any]]] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        top_p: float = 1.0,
        temperature: float = 1.0,
        response_format: str = "auto",
        assistant_id: Optional[str] = None,
    ) -> ent_validator.AssistantRead:
        """
        Create an assistant.

        `tools`            -> relationship / DB tool_configs (unchanged)
        `platform_tools`   -> new inline tool-spec list
        """
        assistant_data = {
            "id": assistant_id,
            "name": name,
            "description": description,
            "model": model,
            "instructions": instructions,
            "tools": tools,
            "platform_tools": platform_tools,
            "meta_data": meta_data,
            "top_p": top_p,
            "temperature": temperature,
            "response_format": response_format,
        }

        try:
            validated = ent_validator.AssistantCreate(**assistant_data)
            logging_utility.info("Creating assistant name=%s model=%s", name, model)

            resp = self._request_with_retries(
                "POST", "/v1/assistants", json=validated.model_dump()
            )
            created = self._parse_response(resp)

            validated_resp = ent_validator.AssistantRead(**created)
            logging_utility.info("Assistant created with id=%s", validated_resp.id)
            return validated_resp
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise AssistantsClientError(f"Validation error: {e}") from e

    def retrieve_assistant(self, assistant_id: str) -> ent_validator.AssistantRead:
        logging_utility.info("Retrieving assistant id=%s", assistant_id)
        try:
            resp = self._request_with_retries("GET", f"/v1/assistants/{assistant_id}")
            data = self._parse_response(resp)
            return ent_validator.AssistantRead(**data)
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise AssistantsClientError(f"Validation error: {e}") from e

    def update_assistant(
        self, assistant_id: str, **updates
    ) -> ent_validator.AssistantRead:
        logging_utility.info("Updating assistant id=%s", assistant_id)

        # Never allow primary key overwrite
        updates.pop("id", None)
        updates.pop("assistant_id", None)

        try:
            validated_updates = ent_validator.AssistantUpdate(**updates)
            resp = self._request_with_retries(
                "PUT",
                f"/v1/assistants/{assistant_id}",
                json=validated_updates.model_dump(exclude_unset=True),
            )
            data = self._parse_response(resp)
            return ent_validator.AssistantRead(**data)
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise AssistantsClientError(f"Validation error: {e}") from e

    def delete_assistant(self, assistant_id: str) -> Dict[str, Any]:
        logging_utility.info("Deleting assistant id=%s", assistant_id)
        resp = self._request_with_retries("DELETE", f"/v1/assistants/{assistant_id}")
        return self._parse_response(resp)

    # ------------------------------------------------------------------ #
    #  USER ASSOCIATIONS
    # ------------------------------------------------------------------ #
    def associate_assistant_with_user(
        self, user_id: str, assistant_id: str
    ) -> Dict[str, Any]:
        logging_utility.info("Link assistant %s → user %s", assistant_id, user_id)
        self._request_with_retries(
            "POST", f"/v1/users/{user_id}/assistants/{assistant_id}"
        )
        return {"message": "Assistant associated with user successfully"}

    def disassociate_assistant_from_user(
        self, user_id: str, assistant_id: str
    ) -> Dict[str, Any]:
        logging_utility.info("Unlink assistant %s ← user %s", assistant_id, user_id)
        self._request_with_retries(
            "DELETE", f"/v1/users/{user_id}/assistants/{assistant_id}"
        )
        return {"message": "Assistant disassociated from user successfully"}

    def list_assistants_by_user(
        self, user_id: str
    ) -> List[ent_validator.AssistantRead]:
        logging_utility.info("Listing assistants for user id=%s", user_id)
        try:
            resp = self._request_with_retries("GET", f"/v1/users/{user_id}/assistants")
            raw_list = self._parse_response(resp)
            return [ent_validator.AssistantRead(**a) for a in raw_list]
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise AssistantsClientError(f"Validation error: {e}") from e
