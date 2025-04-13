#!
import os
import time  # Import the time module for timing
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

from projectdavid.clients.base_client import BaseAPIClient

# --- Setup ---
validation = ValidationInterface()
logging_utility = UtilsInterface.LoggingUtility()
load_dotenv()

# --- Constants for Timeout ---
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0


from projectdavid.clients.base_client import BaseAPIClient


class ActionsClient(BaseAPIClient):

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
        ActionsClient inherits from BaseAPIClient.
        Handles API key injection and timeout config using common base.
        """
        super().__init__(
            base_url=base_url
            or os.getenv("ENTITIES_BASE_URL", "http://localhost:9000/"),
            api_key=api_key or os.getenv("ENTITIES_API_KEY"),
            timeout=timeout,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
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
            ).dict()  # Assuming Pydantic v1, use .model_dump(mode="json") for v2+ if needed

            logging_utility.debug(
                "[CreateAction] Payload for action %s: %s", action_id, payload
            )

            response = self.client.post(
                "/v1/actions", json=payload
            )  # This uses the default timeout
            logging_utility.debug(
                "[CreateAction] Response Status Code for %s: %s",
                action_id,
                response.status_code,
            )
            logging_utility.debug(
                "[CreateAction] Response Body for %s: %s",
                action_id,
                response.text[:500],
            )  # Log truncated body
            response.raise_for_status()

            response_data = response.json()
            validated_action = validation.ActionRead(**response_data)
            logging_utility.info(
                "[CreateAction] Action %s created successfully for run %s.",
                action_id,
                run_id,
            )
            return validated_action

        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "[CreateAction] HTTP error creating action for run %s: %s - Response: %s",
                run_id,
                str(e),
                e.response.text[:500],
            )
            raise ValueError(f"HTTP error during action creation: {str(e)}")
        except httpx.TimeoutException as e:  # Catch potential timeout here too
            logging_utility.error(
                "[CreateAction] Timeout creating action for run %s: %s", run_id, str(e)
            )
            raise TimeoutError(f"Timeout during action creation: {str(e)}") from e
        except httpx.RequestError as e:  # Catch other request errors
            logging_utility.error(
                "[CreateAction] Request error creating action for run %s: %s",
                run_id,
                str(e),
            )
            raise ConnectionError(
                f"Request error during action creation: {str(e)}"
            ) from e
        except Exception as e:
            logging_utility.error(
                "[CreateAction] Unexpected error creating action for run %s: %s",
                run_id,
                str(e),
                exc_info=True,
            )
            raise ValueError(f"Unexpected error during action creation: {str(e)}")

    def get_action(self, action_id: str) -> validation.ActionRead:
        """
        Retrieve a specific action by its ID.
        """
        # --- NO CHANGES IN THIS METHOD (beyond more specific error logging) ---
        try:
            logging_utility.debug(
                "[GetAction] Retrieving action with ID: %s", action_id
            )
            response = self.client.get(
                f"/v1/actions/{action_id}"
            )  # This uses the default timeout
            response.raise_for_status()
            response_data = response.json()
            validated_action = validation.ActionRead(**response_data)
            logging_utility.info(
                "[GetAction] Action %s retrieved successfully.", action_id
            )
            logging_utility.debug(
                "[GetAction] Validated action %s data: %s",
                action_id,
                validated_action.dict(),  # Use model_dump for v2+
            )
            return validated_action

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                error_msg = f"[GetAction] Action {action_id} not found: {str(e)}"
                logging_utility.error(error_msg)
                raise ValueError(error_msg)  # Keep specific 404 error
            else:
                logging_utility.error(
                    "[GetAction] HTTP error retrieving action %s: %s - Response: %s",
                    action_id,
                    str(e),
                    e.response.text[:500],
                )
                raise ValueError(f"HTTP error retrieving action {action_id}: {str(e)}")
        except httpx.TimeoutException as e:  # Catch potential timeout
            logging_utility.error(
                "[GetAction] Timeout retrieving action %s: %s", action_id, str(e)
            )
            raise TimeoutError(
                f"Timeout retrieving action {action_id}: {str(e)}"
            ) from e
        except httpx.RequestError as e:  # Catch other request errors
            logging_utility.error(
                "[GetAction] Request error retrieving action %s: %s", action_id, str(e)
            )
            raise ConnectionError(
                f"Request error retrieving action {action_id}: {str(e)}"
            ) from e
        except ValidationError as e:
            logging_utility.error(
                "[GetAction] Response validation failed for action %s: %s",
                action_id,
                str(e),
            )
            raise ValueError(f"Invalid action data format for {action_id}: {str(e)}")
        except Exception as e:
            logging_utility.error(
                "[GetAction] Unexpected error retrieving action %s: %s",
                action_id,
                str(e),
                exc_info=True,
            )
            raise ValueError(
                f"Unexpected error retrieving action {action_id}: {str(e)}"
            )

    def update_action(
        self,
        action_id: str,
        status: validation.ActionStatus,
        result: Optional[Dict[str, Any]] = None,
    ) -> validation.ActionRead:
        """Update an action's status and result."""
        # --- NO CHANGES IN THIS METHOD (beyond more specific error logging) ---
        try:
            payload = validation.ActionUpdate(status=status, result=result).dict(
                exclude_none=True  # Assuming Pydantic v1, use model_dump(mode="json", exclude_unset=True) for v2+
            )
            logging_utility.debug(
                "[UpdateAction] Payload for action update %s: %s", action_id, payload
            )
            response = self.client.put(
                f"/v1/actions/{action_id}", json=payload
            )  # This uses the default timeout
            response.raise_for_status()
            response_data = response.json()
            validated_action = validation.ActionRead(**response_data)
            logging_utility.info(
                "[UpdateAction] Action %s updated successfully.", action_id
            )
            return validated_action

        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "[UpdateAction] HTTP error updating action %s: %s - Response: %s",
                action_id,
                str(e),
                e.response.text[:500],
            )
            raise ValueError(
                f"HTTP error during action update for {action_id}: {str(e)}"
            )
        except httpx.TimeoutException as e:  # Catch potential timeout
            logging_utility.error(
                "[UpdateAction] Timeout updating action %s: %s", action_id, str(e)
            )
            raise TimeoutError(f"Timeout updating action {action_id}: {str(e)}") from e
        except httpx.RequestError as e:  # Catch other request errors
            logging_utility.error(
                "[UpdateAction] Request error updating action %s: %s", action_id, str(e)
            )
            raise ConnectionError(
                f"Request error updating action {action_id}: {str(e)}"
            ) from e
        except Exception as e:
            logging_utility.error(
                "[UpdateAction] Unexpected error updating action %s: %s",
                action_id,
                str(e),
                exc_info=True,
            )
            raise ValueError(
                f"Unexpected error during action update for {action_id}: {str(e)}"
            )

    def get_actions_by_status(
        self, run_id: str, status: str = "pending"
    ) -> List[Dict[str, Any]]:
        """Retrieve actions by run_id and status."""
        # --- NO CHANGES IN THIS METHOD (beyond more specific error logging) ---
        try:
            logging_utility.debug(
                "[GetActionsByStatus] Retrieving actions for run_id: %s with status: %s",
                run_id,
                status or "not specified",
            )
            response = self.client.get(  # This uses the default timeout
                f"/v1/runs/{run_id}/actions/status", params={"status": status}
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                logging_utility.error(
                    "[GetActionsByStatus] Unexpected content type '%s' for run %s, status '%s'. Body: %s",
                    content_type,
                    run_id,
                    status,
                    response.text[:500],
                )
                raise ValueError(f"Unexpected content type: {content_type}")

            response_data = response.json()
            logging_utility.info(
                "[GetActionsByStatus] %d action(s) with status '%s' retrieved successfully for run_id: %s",
                len(response_data),
                status,
                run_id,
            )
            return response_data
        except httpx.TimeoutException as e:  # Catch potential timeout
            logging_utility.error(
                "[GetActionsByStatus] Timeout retrieving actions for run %s, status '%s': %s",
                run_id,
                status,
                str(e),
            )
            raise TimeoutError(
                f"Timeout retrieving actions by status for run {run_id}: {str(e)}"
            ) from e
        except httpx.RequestError as e:  # Catch other request errors
            logging_utility.error(
                "[GetActionsByStatus] Request error retrieving actions for run %s, status '%s': %s",
                run_id,
                status,
                str(e),
            )
            raise ConnectionError(
                f"Request error retrieving actions by status for {run_id}: {str(e)}"
            ) from e
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "[GetActionsByStatus] HTTP error retrieving actions for run %s, status '%s': %s - Response: %s",
                run_id,
                status,
                str(e),
                e.response.text[:500],
            )
            raise ValueError(
                f"HTTP error during actions retrieval by status for {run_id}: {str(e)}"
            )
        except Exception as e:
            logging_utility.error(
                "[GetActionsByStatus] Unexpected error retrieving actions for run %s, status '%s': %s",
                run_id,
                status,
                str(e),
                exc_info=True,
            )
            raise ValueError(
                f"Unexpected error during actions retrieval by status for {run_id}: {str(e)}"
            )

    def get_pending_actions(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all pending actions for a given run_id.
        Includes timeout handling and timing logs.
        """
        start_time = time.monotonic()  # Start timer
        try:
            logging_utility.debug(
                "[GetPendingActions] Retrieving pending actions for run_id: %s", run_id
            )
            url = f"/v1/actions/pending/{run_id}"

            # --- Make the request using the client with configured timeout ---
            response = self.client.get(url)

            # --- Check for HTTP errors AFTER the request completed (or timed out) ---
            response.raise_for_status()

            # --- Process successful response ---
            response_data = response.json()
            elapsed_time = time.monotonic() - start_time  # End timer
            logging_utility.info(
                "[GetPendingActions] %d pending action(s) retrieved successfully for run %s in %.2f seconds.",
                len(response_data),
                run_id,
                elapsed_time,
            )
            return response_data

        # --- Specific Exception Handling for THIS method ---
        except httpx.TimeoutException as e:
            elapsed_time = time.monotonic() - start_time  # End timer even on timeout
            # Log specific timeout error
            logging_utility.error(
                "[GetPendingActions] Timeout after %.2f seconds retrieving pending actions for run %s: %s",
                elapsed_time,
                run_id,
                str(e),
            )
            # Re-raise as a standard TimeoutError for the monitor to catch
            raise TimeoutError(
                f"Timeout retrieving pending actions for run {run_id}"
            ) from e

        except httpx.HTTPStatusError as e:
            elapsed_time = time.monotonic() - start_time
            # Log HTTP error details
            logging_utility.error(
                "[GetPendingActions] HTTP error %s after %.2f seconds retrieving pending actions for run %s: %s - Response: %s",
                e.response.status_code,
                elapsed_time,
                run_id,
                str(e),
                e.response.text[:500],  # Log truncated body
            )
            # Re-raise as ValueError for the monitor
            raise ValueError(
                f"HTTP error {e.response.status_code} during pending actions retrieval for {run_id}"
            ) from e

        except httpx.RequestError as e:  # Catch other request/network errors
            elapsed_time = time.monotonic() - start_time
            logging_utility.error(
                "[GetPendingActions] Request error after %.2f seconds retrieving pending actions for run %s: %s",
                elapsed_time,
                run_id,
                str(e),
            )
            raise ConnectionError(
                f"Request error retrieving pending actions for {run_id}"
            ) from e

        except Exception as e:  # Catch any other unexpected errors (e.g., JSON decode)
            elapsed_time = time.monotonic() - start_time
            logging_utility.error(
                "[GetPendingActions] Unexpected error after %.2f seconds retrieving pending actions for run %s: %s",
                elapsed_time,
                run_id,
                str(e),
                exc_info=True,
            )
            raise ValueError(
                f"Unexpected error during pending actions retrieval for {run_id}: {str(e)}"
            ) from e
