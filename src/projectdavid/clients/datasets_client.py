import os
from typing import Dict, Optional, Union

from projectdavid_common import UtilsInterface, ValidationInterface

from projectdavid.clients.base_client import BaseAPIClient
from projectdavid.clients.files_client import FileClient

logging_utility = UtilsInterface.LoggingUtility()
validator = ValidationInterface()


class DatasetsClient(BaseAPIClient):
    """
    Client for the training service /v1/datasets endpoints.

    Inherits from BaseAPIClient to ensure consistent authentication
    and session management.
    """

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

        self._file_client = FileClient(base_url=base_url, api_key=api_key)

    def _get_url(self, path: str) -> str:
        """Helper to construct the full training service URL."""
        return f"{self.training_url}{path}"

    # ------------------------------------------------------------------
    # CREATE — upload file then register dataset
    # ------------------------------------------------------------------

    def create(
        self,
        file_path: str,
        name: str,
        fmt: str,
        description: Optional[str] = None,
    ) -> validator.DatasetRead:
        """
        Upload a dataset file and register it with the training service.
        """
        # Step 1 — upload to core API
        logging_utility.info("Uploading dataset file: %s", file_path)
        file_response = self._file_client.upload_file(file_path, purpose="training")
        logging_utility.info("File uploaded — file_id=%s", file_response.id)

        # Step 2 — register with training API
        payload = {
            "name": name,
            "format": fmt,
            "file_id": file_response.id,
            "description": description,
            "filename": file_path.split("/")[-1].split("\\")[-1],
        }

        # Use self.client which already contains the auth headers and timeout config
        response = self.client.post(self._get_url("/v1/datasets/"), json=payload)
        response.raise_for_status()

        dataset = validator.DatasetRead.model_validate(response.json())
        logging_utility.info("Dataset registered — id=%s", dataset.id)
        return dataset

    # ------------------------------------------------------------------
    # PREPARE
    # ------------------------------------------------------------------

    def prepare(self, dataset_id: str) -> dict:
        response = self.client.post(self._get_url(f"/v1/datasets/{dataset_id}/prepare"))
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # RETRIEVE
    # ------------------------------------------------------------------

    def retrieve(self, dataset_id: str) -> validator.DatasetRead:
        response = self.client.get(self._get_url(f"/v1/datasets/{dataset_id}"))
        response.raise_for_status()
        return validator.DatasetRead.model_validate(response.json())

    # ------------------------------------------------------------------
    # LIST
    # ------------------------------------------------------------------

    def list(
        self, status: Optional[str] = None, limit: int = 50
    ) -> validator.DatasetList:
        params: Dict[str, Union[str, int]] = {"limit": limit}
        if status:
            params["status"] = status

        response = self.client.get(self._get_url("/v1/datasets/"), params=params)
        response.raise_for_status()
        return validator.DatasetList.model_validate(response.json())

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def delete(self, dataset_id: str) -> validator.DatasetDeleted:
        response = self.client.delete(self._get_url(f"/v1/datasets/{dataset_id}"))
        response.raise_for_status()
        return validator.DatasetDeleted.model_validate(response.json())
