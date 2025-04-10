import time
from typing import Any, Dict, List, Optional

import httpx
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

ent_validator = ValidationInterface()
logging_utility = UtilsInterface.LoggingUtility()


class RunsClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the RunsClient with the given base URL and optional API key.

        Args:
            base_url (str): The base URL for the runs service.
            api_key (Optional[str]): The API key for authentication.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        logging_utility.info("RunsClient initialized with base_url: %s", self.base_url)

    def create_run(
        self,
        assistant_id: str,
        thread_id: str,
        instructions: Optional[str] = "",
        meta_data: Optional[Dict[str, Any]] = {},
    ) -> ent_validator.Run:
        """
        Create a new run using the provided assistant_id, thread_id, and instructions.
        Returns a Run Pydantic model.

        Args:
            assistant_id (str): The assistant's ID.
            thread_id (str): The thread's ID.
            instructions (Optional[str]): Instructions for the run.
            meta_data (Optional[Dict[str, Any]]): Additional metadata.

        Returns:
            Run: The created run.
        """
        run_data = ent_validator.Run(
            id=UtilsInterface.IdentifierService.generate_run_id(),
            assistant_id=assistant_id,
            thread_id=thread_id,
            instructions=instructions,
            meta_data=meta_data,
            cancelled_at=None,
            completed_at=None,
            created_at=int(time.time()),
            expires_at=int(time.time()) + 3600,  # 1 hour later
            failed_at=None,
            incomplete_details=None,
            last_error=None,
            max_completion_tokens=1000,
            max_prompt_tokens=500,
            model="llama3.1",
            object="run",
            parallel_tool_calls=False,
            required_action=None,
            response_format="text",
            started_at=None,
            status="pending",
            tool_choice="none",
            tools=[],
            truncation_strategy={},
            usage=None,
            temperature=0.7,
            top_p=0.9,
            tool_resources={},
        )

        logging_utility.info(
            "Creating run for assistant_id: %s, thread_id: %s", assistant_id, thread_id
        )
        logging_utility.debug("Run data: %s", run_data.dict())

        try:
            response = self.client.post("/v1/runs", json=run_data.dict())
            response.raise_for_status()
            created_run_data = response.json()

            # Validate the response using the Run model
            validated_run = ent_validator.Run(**created_run_data)
            logging_utility.info(
                "Run created successfully with id: %s", validated_run.id
            )
            return validated_run

        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error occurred while creating run: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("An error occurred while creating run: %s", str(e))
            raise

    def retrieve_run(self, run_id: str) -> ent_validator.RunReadDetailed:
        """
        Retrieve a run by its ID and return it as a RunReadDetailed Pydantic model.

        Args:
            run_id (str): The run ID.

        Returns:
            RunReadDetailed: The retrieved run details.
        """
        logging_utility.info("Retrieving run with id: %s", run_id)
        try:
            response = self.client.get(f"/v1/runs/{run_id}")
            response.raise_for_status()
            run_data = response.json()
            validated_run = ent_validator.RunReadDetailed(**run_data)
            logging_utility.info(
                "Run with id %s retrieved and validated successfully", run_id
            )
            return validated_run

        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Data validation failed: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while retrieving run: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An unexpected error occurred while retrieving run: %s", str(e)
            )
            raise

    def update_run_status(self, run_id: str, new_status: str) -> ent_validator.Run:
        """
        Update the status of a run.

        Args:
            run_id (str): The run ID.
            new_status (str): The new status to set.

        Returns:
            Run: The updated run.
        """
        logging_utility.info(
            "Updating run status for run_id: %s to %s", run_id, new_status
        )
        update_data = {"status": new_status}

        try:
            validated_data = ent_validator.RunStatusUpdate(**update_data)
            response = self.client.put(
                f"/v1/runs/{run_id}/status", json=validated_data.dict()
            )
            response.raise_for_status()

            updated_run = response.json()
            validated_run = ent_validator.Run(**updated_run)
            logging_utility.info("Run status updated successfully")
            return validated_run

        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while updating run status: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while updating run status: %s", str(e)
            )
            raise

    def list_runs(self, limit: int = 20, order: str = "asc") -> List[ent_validator.Run]:
        """
        List runs with the given limit and order.

        Args:
            limit (int): Maximum number of runs to retrieve.
            order (str): 'asc' or 'desc' for ordering.

        Returns:
            List[Run]: A list of runs.
        """
        logging_utility.info("Listing runs with limit: %d, order: %s", limit, order)
        params = {"limit": limit, "order": order}
        try:
            response = self.client.get("/v1/runs", params=params)
            response.raise_for_status()
            runs = response.json()
            validated_runs = [ent_validator.Run(**run) for run in runs]
            logging_utility.info("Retrieved %d runs", len(validated_runs))
            return validated_runs
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error occurred while listing runs: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("An error occurred while listing runs: %s", str(e))
            raise

    def delete_run(self, run_id: str) -> Dict[str, Any]:
        """
        Delete a run by its ID.

        Args:
            run_id (str): The run ID.

        Returns:
            Dict[str, Any]: The deletion result.
        """
        logging_utility.info("Deleting run with id: %s", run_id)
        try:
            response = self.client.delete(f"/v1/runs/{run_id}")
            response.raise_for_status()
            result = response.json()
            logging_utility.info("Run deleted successfully")
            return result
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error occurred while deleting run: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("An error occurred while deleting run: %s", str(e))
            raise

    def generate(
        self, run_id: str, model: str, prompt: str, stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate content for a run based on the provided model and prompt.

        Args:
            run_id (str): The run ID.
            model (str): The model to use.
            prompt (str): The prompt text.
            stream (bool): Whether to stream the response.

        Returns:
            Dict[str, Any]: The generated content.
        """
        logging_utility.info(
            "Generating content for run_id: %s, model: %s", run_id, model
        )
        try:
            run = self.retrieve_run(run_id)
            response = self.client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": stream,
                    "context": run.meta_data.get("context", []),
                    "temperature": run.temperature,
                    "top_p": run.top_p,
                },
            )
            response.raise_for_status()
            result = response.json()
            logging_utility.info("Content generated successfully")
            return result
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while generating content: %s", str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while generating content: %s", str(e)
            )
            raise

    def chat(
        self,
        run_id: str,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Chat using a run, model, and provided messages.

        Args:
            run_id (str): The run ID.
            model (str): The model to use.
            messages (List[Dict[str, Any]]): The messages for context.
            stream (bool): Whether to stream the response.

        Returns:
            Dict[str, Any]: The chat response.
        """
        logging_utility.info("Chatting for run_id: %s, model: %s", run_id, model)
        try:
            run = self.retrieve_run(run_id)
            response = self.client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    "context": run.meta_data.get("context", []),
                    "temperature": run.temperature,
                    "top_p": run.top_p,
                },
            )
            response.raise_for_status()
            result = response.json()
            logging_utility.info("Chat completed successfully")
            return result
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error occurred during chat: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("An error occurred during chat: %s", str(e))
            raise

    def cancel_run(self, run_id: str) -> Dict[str, Any]:
        """
        Cancel a run by its ID.

        Args:
            run_id (str): The run ID.

        Returns:
            Dict[str, Any]: The cancellation result.
        """
        logging_utility.info("Cancelling run with id: %s", run_id)
        try:
            response = self.client.post(f"/v1/runs/{run_id}/cancel")
            response.raise_for_status()
            result = response.json()
            logging_utility.info("Run %s cancelled successfully", run_id)
            return result
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while cancelling run %s: %s", run_id, str(e)
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An error occurred while cancelling run %s: %s", run_id, str(e)
            )
            raise
