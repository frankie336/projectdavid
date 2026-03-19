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
        self.training_url = (
            training_url or os.getenv("TRAINING_BASE_URL", "http://localhost:9001")
        ).rstrip("/")

    def create(
        self,
        dataset_id: str,
        base_model: str,
        framework: str = "unsloth",
        config: dict = None,
    ) -> validator.TrainingJobRead:
        payload = {
            "dataset_id": dataset_id,
            "base_model": base_model,
            "framework": framework,
            "config": config or {},
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
