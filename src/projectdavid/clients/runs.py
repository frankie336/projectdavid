import json  # Added for parsing response text in errors potentially
import time
from enum import Enum  # Added to use StatusEnum type hint if possible
from typing import Any, Dict, List, Optional

import httpx
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.schemas.enums import StatusEnum
from pydantic import ValidationError

# --- Initialization ---
ent_validator = ValidationInterface()
logging_utility = UtilsInterface.LoggingUtility()
# --- End Initialization ---


class RunsClient:
    """
    Client for interacting with the Runs endpoints of the ProjectDavid API.
    Provides methods for creating, retrieving, updating, listing, deleting,
    cancelling runs, and waiting for specific run states.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the RunsClient.

        Args:
            base_url (str): The base URL for the ProjectDavid API runs service.
            api_key (Optional[str]): The API key for authentication.
        """
        if not base_url:
            raise ValueError("base_url must be provided.")
        self.base_url = base_url
        self.api_key = api_key
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0,  # Default timeout for requests
        )
        logging_utility.info("RunsClient initialized with base_url: %s", self.base_url)

    def close(self):
        """Closes the underlying HTTP client session."""
        if hasattr(self, "client") and self.client and not self.client.is_closed:
            try:
                self.client.close()
                logging_utility.info("RunsClient httpx client closed.")
            except Exception as e:
                logging_utility.error(f"Error closing RunsClient httpx client: {e}")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, ensuring client is closed."""
        self.close()

    # --- Core CRUD and Action Methods ---

    def create_run(
        self,
        assistant_id: str,
        thread_id: str,
        instructions: Optional[str] = "",
        meta_data: Optional[Dict[str, Any]] = None,  # Default to None
        **kwargs,  # Allow passing other Run parameters if API supports them
    ) -> ent_validator.Run:
        """
        Create a new run for an assistant on a thread.

        Args:
            assistant_id (str): The assistant's ID.
            thread_id (str): The thread's ID.
            instructions (Optional[str]): Override instructions for this specific run.
            meta_data (Optional[Dict[str, Any]]): Metadata for the run.
            **kwargs: Additional parameters matching the Run model (e.g., model, tools).

        Returns:
            ent_validator.Run: The created run object.
        """
        # Prepare base run data
        run_payload_dict = {
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "instructions": instructions or None,
            "meta_data": meta_data or {},
            **kwargs,
        }
        # Remove None values if API doesn't expect them
        run_payload = {k: v for k, v in run_payload_dict.items() if v is not None}

        logging_utility.info(
            "Creating run for assistant_id: %s, thread_id: %s", assistant_id, thread_id
        )
        logging_utility.debug("Run create payload: %s", run_payload)

        try:
            response = self.client.post("/v1/runs", json=run_payload)
            response.raise_for_status()
            created_run_data = response.json()
            validated_run = ent_validator.Run(**created_run_data)
            logging_utility.info(
                "Run created successfully with id: %s", validated_run.id
            )
            return validated_run

        except ValidationError as e:
            logging_utility.error(
                "Validation error preparing run creation data: %s", e.json()
            )
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while creating run: %s - %s",
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while creating run: %s", str(e), exc_info=True
            )
            raise

    def retrieve_run(self, run_id: str) -> ent_validator.RunReadDetailed:
        """
        Retrieve detailed information about a specific run.

        Args:
            run_id (str): The unique ID of the run.

        Returns:
            ent_validator.RunReadDetailed: The detailed run object.
        """
        logging_utility.info("Retrieving run with id: %s", run_id)
        try:
            response = self.client.get(f"/v1/runs/{run_id}")
            response.raise_for_status()
            run_data = response.json()
            validated_run = ent_validator.RunReadDetailed(**run_data)
            logging_utility.info("Run with id %s retrieved successfully", run_id)
            return validated_run

        except ValidationError as e:
            logging_utility.error(
                "Validation error processing retrieved run %s: %s", run_id, e.json()
            )
            raise ValueError(f"Data validation failed: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while retrieving run %s: %s - %s",
                run_id,
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An unexpected error occurred while retrieving run %s: %s",
                run_id,
                str(e),
                exc_info=True,
            )
            raise

    def update_run_status(self, run_id: str, new_status: str) -> ent_validator.Run:
        """
        Update the status of an existing run.

        Args:
            run_id (str): The ID of the run to update.
            new_status (str): The new status string (e.g., 'processing', 'completed').

        Returns:
            ent_validator.Run: The updated run object.
        """
        logging_utility.info(
            "Updating run status for run_id: %s to %s", run_id, new_status
        )
        update_data = {"status": new_status}

        try:
            # Validate the status update payload (assumes RunStatusUpdate exists)
            validated_data = ent_validator.RunStatusUpdate(**update_data)
            response = self.client.put(
                f"/v1/runs/{run_id}/status", json=validated_data.model_dump()
            )
            response.raise_for_status()

            updated_run = response.json()
            validated_run = ent_validator.Run(**updated_run)
            logging_utility.info(
                f"Run {run_id} status updated successfully to {new_status}"
            )
            return validated_run

        except ValidationError as e:
            logging_utility.error(
                "Validation error preparing run status update: %s", e.json()
            )
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while updating run status %s: %s - %s",
                run_id,
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while updating run status %s: %s",
                run_id,
                str(e),
                exc_info=True,
            )
            raise

    def list_runs(
        self, limit: int = 20, order: str = "desc", **params
    ) -> List[ent_validator.Run]:
        """
        List runs, optionally filtered and ordered.

        Args:
            limit (int): Maximum number of runs to retrieve (default: 20).
            order (str): 'asc' or 'desc' for ordering by creation time (default: 'desc').
            **params: Additional query parameters for filtering (e.g., thread_id).

        Returns:
            List[ent_validator.Run]: A list of run objects.
        """
        logging_utility.info(
            "Listing runs with limit: %d, order: %s, params: %s", limit, order, params
        )
        query_params = {"limit": limit, "order": order, **params}
        try:
            response = self.client.get("/v1/runs", params=query_params)
            response.raise_for_status()
            runs = response.json()
            if not isinstance(runs, list):
                logging_utility.error(
                    f"API response for listing runs is not a list: {runs}"
                )
                raise ValueError("Invalid API response format: expected a list.")

            validated_runs = [ent_validator.Run(**run) for run in runs]
            logging_utility.info("Retrieved %d runs", len(validated_runs))
            return validated_runs
        except ValidationError as e:
            logging_utility.error(
                "Validation error processing listed runs: %s", e.json()
            )
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while listing runs: %s - %s",
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while listing runs: %s", str(e), exc_info=True
            )
            raise

    def delete_run(self, run_id: str) -> Dict[str, Any]:
        """
        Delete a specific run by its ID.

        Args:
            run_id (str): The ID of the run to delete.

        Returns:
            Dict[str, Any]: A confirmation object (or empty on 204).
        """
        logging_utility.info("Deleting run with id: %s", run_id)
        try:
            response = self.client.delete(f"/v1/runs/{run_id}")
            response.raise_for_status()
            if response.status_code == 204:
                logging_utility.info(
                    "Run %s deleted successfully (204 No Content)", run_id
                )
                return {"id": run_id, "object": "run.deleted", "deleted": True}
            result = response.json()
            logging_utility.info(
                "Run %s deleted successfully (Status: %d)", run_id, response.status_code
            )
            return result
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while deleting run %s: %s - %s",
                run_id,
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while deleting run %s: %s",
                run_id,
                str(e),
                exc_info=True,
            )
            raise

    def cancel_run(self, run_id: str) -> ent_validator.Run:
        """
        Attempt to cancel a run that is currently in progress.

        Args:
            run_id (str): The ID of the run to cancel.

        Returns:
            ent_validator.Run: The run object, potentially with status 'cancelling' or 'cancelled'.
        """
        logging_utility.info("Attempting to cancel run with id: %s", run_id)
        try:
            response = self.client.post(f"/v1/runs/{run_id}/cancel")
            response.raise_for_status()
            result_data = response.json()
            validated_run = ent_validator.Run(**result_data)
            logging_utility.info(
                "Run %s cancellation request accepted (Current status: %s)",
                run_id,
                validated_run.status,
            )
            return validated_run
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while cancelling run %s: %s - %s",
                run_id,
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while cancelling run %s: %s",
                run_id,
                str(e),
                exc_info=True,
            )
            raise

    # --- MODIFIED POLLING HELPER METHOD ---
    def wait_for_action_required(
        self, run_id: str, timeout: float = 60.0, interval: float = 1.0
    ) -> Optional[ent_validator.RunReadDetailed]:
        """
        Polls the status of a specific run until it requires action,
        reaches a terminal state, or the timeout is exceeded.

        Args:
            run_id (str): The ID of the run to monitor.
            timeout (float): Maximum time to wait in seconds. Defaults to 60.
            interval (float): Time to wait between polling attempts in seconds.
                              Defaults to 1.0.

        Returns:
            Optional[RunReadDetailed]: The full Run object if its status becomes
                                       'action_required' within the timeout.
                                       Returns None if the run reaches a terminal
                                       state or if the timeout is reached before
                                       action is required.

        Raises:
            ValueError: If timeout or interval are non-positive.
            httpx.HTTPStatusError: Can be re-raised by underlying retrieve_run if a
                                   persistent API error occurs (e.g., 404 Not Found after retries fail).
            Exception: For unexpected errors during polling.
        """
        if timeout <= 0 or interval <= 0:
            raise ValueError("Timeout and interval must be positive numbers.")

        start_time = time.time()
        logging_utility.info(
            f"Waiting for run {run_id} to require action (timeout: {timeout}s)..."
        )

        # Define terminal states using the exact string values from your StatusEnum
        terminal_states = {
            StatusEnum.completed.value,
            StatusEnum.failed.value,
            StatusEnum.cancelled.value,
            StatusEnum.expired.value,
            # Do not include 'deleted' as it's not a typical run lifecycle state
        }
        # Define states where we should keep polling (exclude terminal & target)
        # Using the provided StatusEnum for accuracy
        transient_states = {
            StatusEnum.queued.value,
            StatusEnum.in_progress.value,
            StatusEnum.processing.value,
            StatusEnum.cancelling.value,
            StatusEnum.pending.value,  # Include if 'pending' is a valid pre-action state
            StatusEnum.retrying.value,
            # Exclude 'active' unless it's confirmed to be transient *before* action_required
        }
        target_state = StatusEnum.pending_action.value  # "action_required"

        while (time.time() - start_time) < timeout:
            try:
                # Use the existing retrieve_run method
                current_run = self.retrieve_run(run_id)
                # Access status; ensure it matches the expected type (string or enum)
                # Using .value assumes retrieve_run returns the enum member or validates to it
                # If retrieve_run returns a simple string, direct comparison is fine.
                # Let's assume it might return the enum member for safety:
                if isinstance(current_run.status, Enum):
                    current_status_str = current_run.status.value
                else:
                    current_status_str = str(current_run.status)  # Fallback to string

                logging_utility.debug(
                    f"Polling run {run_id}: Current status = '{current_status_str}'"
                )

                # Check for target state
                if current_status_str == target_state:
                    logging_utility.info(f"Run {run_id} now requires action.")
                    return current_run

                # Check for terminal states
                if current_status_str in terminal_states:
                    logging_utility.info(
                        f"Run {run_id} reached terminal state '{current_status_str}' while waiting for action."
                    )
                    return None  # Stop polling

                # Check if status is unexpected (neither target, terminal, nor known transient)
                if current_status_str not in transient_states:
                    logging_utility.warning(
                        f"Run {run_id} entered unexpected non-terminal state '{current_status_str}' while waiting for action. Stopping wait."
                    )
                    return None  # Stop polling on unexpected states

                # If in a valid transient state, wait and continue polling
                time.sleep(interval)

            except httpx.HTTPStatusError as e:
                # Handle 404 specifically - Run doesn't exist
                if e.response.status_code == 404:
                    logging_utility.error(
                        f"Run {run_id} not found during polling. Stopping wait."
                    )
                    raise  # Re-raise 404 as it's a definitive state
                else:
                    # Log other HTTP errors but maybe continue polling cautiously? Or break?
                    logging_utility.error(
                        f"HTTP error {e.response.status_code} retrieving run {run_id} during polling: {e.response.text}. Stopping wait."
                    )
                    return None  # Stop polling on other HTTP errors for safety

            except Exception as e:
                # Catch other errors from retrieve_run (network, validation, etc.)
                logging_utility.error(
                    f"Error retrieving run {run_id} during polling: {e}", exc_info=True
                )
                logging_utility.warning(
                    f"Stopping wait for run {run_id} due to error during status retrieval."
                )
                return None  # Indicate polling stopped due to error

        # Timeout reached
        logging_utility.warning(
            f"Timeout reached ({timeout}s) waiting for run {run_id} to require action."
        )
        return None

    # --- Deprecated/Internal Methods Removed ---
    # Removed generate and chat as they seemed like placeholders/examples


# Note: No close method was present in the provided code, added one for completeness.
# Added __enter__ and __exit__ for context manager support.
