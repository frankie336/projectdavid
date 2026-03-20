import os
from typing import Optional

from projectdavid_common import UtilsInterface, ValidationInterface

from projectdavid.clients.base_client import BaseAPIClient

logging_utility = UtilsInterface.LoggingUtility()
validator = ValidationInterface()


class ModelsClient(BaseAPIClient):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        training_url: Optional[str] = None,
    ):
        super().__init__(base_url=base_url, api_key=api_key)
        raw_training_url = training_url or os.getenv(
            "TRAINING_BASE_URL", "http://localhost:9001"
        )
        self.training_url = raw_training_url.rstrip("/")

    def list(self, limit: int = 50, offset: int = 0) -> validator.FineTunedModelList:
        response = self.client.get(
            f"{self.training_url}/v1/fine-tuned-models/",
            params={"limit": limit, "offset": offset},
        )
        response.raise_for_status()
        return validator.FineTunedModelList.model_validate(response.json())

    def retrieve(self, model_id: str) -> validator.FineTunedModelRead:
        response = self.client.get(
            f"{self.training_url}/v1/fine-tuned-models/{model_id}"
        )
        response.raise_for_status()
        return validator.FineTunedModelRead.model_validate(response.json())

    def delete(self, model_id: str) -> validator.FineTunedModelDeleted:
        response = self.client.delete(
            f"{self.training_url}/v1/fine-tuned-models/{model_id}"
        )
        response.raise_for_status()
        return validator.FineTunedModelDeleted.model_validate(response.json())

    def activate(self, model_id: str) -> validator.ActivateModelResponse:
        """
        Set this fine-tuned model as the active instance for inference.
        """
        logging_utility.info("Activating fine-tuned model: %s", model_id)
        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/{model_id}/activate"
        )
        response.raise_for_status()
        return validator.ActivateModelResponse.model_validate(response.json())

    def deactivate_all(self) -> dict:
        """
        Deactivate any currently active fine-tuned model.
        Returns the system to its base model state.
        """
        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/deactivate-all"
        )
        response.raise_for_status()
        return response.json()
