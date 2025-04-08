#!
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

validation = ValidationInterface()
logging_utility = UtilsInterface.LoggingUtility()
load_dotenv()


class ActionsClient:
    def __init__(
        self,
        base_url: str = os.getenv("ASSISTANTS_BASE_URL", "http://localhost:9000/"),
        api_key: Optional[str] = None,
    ):
        """
        Initialize with base URL and API key for authentication.
        """
        self.base_url = base_url
        self.api_key = api_key or os.getenv("API_KEY", "your_api_key")
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        logging_utility.info(
            "ActionsClient initialized with base_url: %s", self.base_url
        )

    def create_action(
        self,
        tool_name: str,
        run_id: str,
        function_args: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> validation.ActionRead:
        """Create a new action using the provided tool_name, run_id, and function_args."""
        try:
            action_id = UtilsInterface.IdentifierService.generate_action_id()
            expires_at_iso = expires_at.isoformat() if expires_at else None

            payload = validation.ActionCreate(
                id=action_id,
                tool_name=tool_name,
                run_id=run_id,
                function_args=function_args or {},
                expires_at=expires_at_iso,
                status="pending",
            ).dict()

            logging_utility.debug("Payload for action creation: %s", payload)

            response = self.client.post("/v1/actions", json=payload)
            logging_utility.debug("Response Status Code: %s", response.status_code)
            logging_utility.debug("Response Body: %s", response.text)
            response.raise_for_status()

            response_data = response.json()
            validated_action = validation.ActionRead(**response_data)
            logging_utility.info("Action created successfully with ID: %s", action_id)
            return validated_action

        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error during action creation: %s", str(e))
            raise ValueError(f"HTTP error during action creation: {str(e)}")
        except Exception as e:
            logging_utility.error("Unexpected error during action creation: %s", str(e))
            raise ValueError(f"Unexpected error: {str(e)}")

    def get_action(self, action_id: str) -> validation.ActionRead:
        """
        Retrieve a specific action by its ID.
        """
        try:
            logging_utility.debug("Retrieving action with ID: %s", action_id)
            response = self.client.get(f"/v1/actions/{action_id}")
            response.raise_for_status()
            response_data = response.json()
            validated_action = validation.ActionRead(**response_data)
            logging_utility.info("Action retrieved successfully with ID: %s", action_id)
            logging_utility.debug(
                "Validated action data: %s", validated_action.model_dump(mode="json")
            )
            return validated_action

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                error_msg = f"Action {action_id} not found: {str(e)}"
                logging_utility.error(error_msg)
                raise ValueError(error_msg)
            logging_utility.error("HTTP error during action retrieval: %s", str(e))
            raise ValueError(f"HTTP error during action retrieval: {str(e)}")
        except ValidationError as e:
            logging_utility.error("Response validation failed: %s", str(e))
            raise ValueError(f"Invalid action data format: {str(e)}")
        except httpx.RequestError as e:
            error_msg = f"Request error: {str(e)}"
            logging_utility.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logging_utility.error("Unexpected error: %s", str(e))
            raise ValueError(f"Unexpected error: {str(e)}")

    def update_action(
        self,
        action_id: str,
        status: validation.ActionStatus,
        result: Optional[Dict[str, Any]] = None,
    ) -> validation.ActionRead:
        """Update an action's status and result."""
        try:
            payload = validation.ActionUpdate(status=status, result=result).dict(
                exclude_none=True
            )
            logging_utility.debug("Payload for action update: %s", payload)
            response = self.client.put(f"/v1/actions/{action_id}", json=payload)
            response.raise_for_status()
            response_data = response.json()
            validated_action = validation.ActionRead(**response_data)
            logging_utility.info("Action updated successfully with ID: %s", action_id)
            return validated_action

        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error during action update: %s", str(e))
            raise ValueError(f"HTTP error during action update: {str(e)}")

    def get_actions_by_status(
        self, run_id: str, status: str = "pending"
    ) -> List[Dict[str, Any]]:
        """Retrieve actions by run_id and status."""
        try:
            logging_utility.debug(
                "Retrieving actions for run_id: %s with status: %s",
                run_id,
                status or "not specified",
            )
            response = self.client.get(
                f"/v1/runs/{run_id}/actions/status", params={"status": status}
            )
            response.raise_for_status()
            if response.headers.get("Content-Type") == "application/json":
                response_data = response.json()
            else:
                logging_utility.error(
                    "Unexpected content type: %s", response.headers.get("Content-Type")
                )
                raise ValueError(
                    f"Unexpected content type: {response.headers.get('Content-Type')}"
                )
            logging_utility.info(
                "Actions retrieved successfully for run_id: %s with status: %s",
                run_id,
                status,
            )
            return response_data
        except httpx.RequestError as e:
            logging_utility.error(
                "Error requesting actions for run_id %s: %s", run_id, str(e)
            )
            raise ValueError(f"Request error: {str(e)}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error during actions retrieval for run_id %s with status %s: %s",
                run_id,
                status,
                str(e),
            )
            raise ValueError(f"HTTP error during actions retrieval: {str(e)}")

    def get_pending_actions(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all pending actions for a given run_id.
        """
        try:
            logging_utility.debug("Retrieving pending actions with run_id: %s", run_id)
            url = f"/v1/actions/pending/{run_id}"
            response = self.client.get(url)
            response.raise_for_status()
            response_data = response.json()
            logging_utility.info("Pending actions retrieved successfully")
            return response_data
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error during pending actions retrieval: %s", str(e)
            )
            raise ValueError(f"HTTP error during pending actions retrieval: {str(e)}")
