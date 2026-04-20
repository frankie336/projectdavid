import os
from typing import Optional

from projectdavid_common import UtilsInterface, ValidationInterface

from projectdavid.clients.base_client import BaseAPIClient

logging_utility = UtilsInterface.LoggingUtility()
validator = ValidationInterface()


class TrainingClient(BaseAPIClient):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        training_url: Optional[str] = None,
    ):
        super().__init__(base_url=base_url, api_key=api_key)

        # Training routes are behind the same nginx proxy as the core API.
        # Use base_url as the default — no separate training_url needed
        # unless explicitly overridden via TRAINING_BASE_URL.
        resolved_url = (
            training_url
            or os.getenv("TRAINING_BASE_URL")
            or base_url
            or "http://localhost:80"
        )
        self.training_url = resolved_url.rstrip("/")

    def create(
        self,
        dataset_id: str,
        base_model: str,
        framework: str = "unsloth",
        config: validator.TrainingConfig | dict | None = None,
    ) -> validator.TrainingJobRead:
        """
        Create and queue a training job.

        `config` accepts either a dict of raw parameters or a TrainingConfig
        instance (recommended — gets IDE autocomplete + local Pydantic
        validation before the HTTP round-trip).
        """
        if isinstance(config, validator.TrainingConfig):
            config_payload = config.model_dump(exclude_none=True)
        else:
            config_payload = config or {}

        payload = {
            "dataset_id": dataset_id,
            "base_model": base_model,
            "framework": framework,
            "config": config_payload,
        }
        response = self.client.post(
            f"{self.training_url}/v1/training-jobs/", json=payload
        )
        response.raise_for_status()
        return validator.TrainingJobRead.model_validate(response.json())

    def retrieve(self, job_id: str) -> validator.TrainingJobRead:
        response = self.client.get(f"{self.training_url}/v1/training-jobs/{job_id}")
        response.raise_for_status()
        return validator.TrainingJobRead.model_validate(response.json())

    def cancel(self, job_id: str) -> validator.TrainingJobCancelResponse:
        """
        Cancel a training job.

        Idempotent — calling cancel on a job in a terminal state (completed,
        failed, cancelled) returns the current status without error.

        For in-progress jobs, cancellation is asynchronous. The response status
        will be 'cancelling'; poll retrieve(job_id) to observe the transition
        to 'cancelled' after the worker unwinds the subprocess (typically
        within 30 seconds).

        Partial training artifacts are discarded on cancellation.
        """
        response = self.client.post(
            f"{self.training_url}/v1/training-jobs/{job_id}/cancel"
        )
        response.raise_for_status()
        return validator.TrainingJobCancelResponse.model_validate(response.json())

    # ------------------------------------------------------------------
    # DIAGNOSTIC PEEK (Secure Multi-tenant Gateway)
    # ------------------------------------------------------------------
    def peek_queue(self) -> validator.TrainingQueueList:
        """
        Securely check the remote Redis queue via the Training API to see
        pending jobs belonging only to the current user.
        """
        response = self.client.get(f"{self.training_url}/v1/training-jobs/queue/peek")
        response.raise_for_status()
        return validator.TrainingQueueList.model_validate(response.json())
