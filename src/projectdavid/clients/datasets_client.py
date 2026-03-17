# projectdavid/clients/datasets_client.py

import os
from typing import Optional

import httpx
from projectdavid_common import UtilsInterface, ValidationInterface

from projectdavid.clients.base_client import BaseAPIClient
from projectdavid.clients.files_client import FileClient

logging_utility = UtilsInterface.LoggingUtility()
validator = ValidationInterface()


class DatasetsClient(BaseAPIClient):
    """
    Client for the training service /v1/datasets endpoints.

    Upload is a two-step process transparent to the caller:
      1. File bytes → core API  (POST /v1/uploads)
      2. file_id + metadata → training API  (POST /v1/datasets/)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        training_url: Optional[str] = None,
    ):
        super().__init__(base_url=base_url, api_key=api_key)
        self.training_url = training_url or os.getenv(
            "TRAINING_BASE_URL", "http://localhost:9001"
        )
        self._file_client = FileClient(base_url=base_url, api_key=api_key)

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

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

        Args:
            file_path: Local path to the JSONL/chatml/alpaca/sharegpt file.
            name:      Human-readable dataset name.
            fmt:       One of: chatml | alpaca | sharegpt | jsonl
            description: Optional description.

        Returns:
            DatasetRead schema.
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

        with httpx.Client(base_url=self.training_url, follow_redirects=True) as client:
            response = client.post(
                "/v1/datasets/",
                json=payload,
                headers=self._auth_headers(),
                timeout=30.0,
            )
            response.raise_for_status()

        dataset = validator.DatasetRead.model_validate(response.json())
        logging_utility.info("Dataset registered — id=%s", dataset.id)
        return dataset

    # ------------------------------------------------------------------
    # PREPARE
    # ------------------------------------------------------------------

    def prepare(self, dataset_id: str) -> dict:
        with httpx.Client(base_url=self.training_url, follow_redirects=True) as client:
            response = client.post(
                f"/v1/datasets/{dataset_id}/prepare",
                headers=self._auth_headers(),
                timeout=30.0,
            )
            response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # RETRIEVE
    # ------------------------------------------------------------------

    def retrieve(self, dataset_id: str) -> validator.DatasetRead:
        with httpx.Client(base_url=self.training_url, follow_redirects=True) as client:
            response = client.get(
                f"/v1/datasets/{dataset_id}",
                headers=self._auth_headers(),
                timeout=30.0,
            )
            response.raise_for_status()
        return validator.DatasetRead.model_validate(response.json())

    # ------------------------------------------------------------------
    # LIST
    # ------------------------------------------------------------------

    def list(
        self, status: Optional[str] = None, limit: int = 50
    ) -> validator.DatasetList:
        params = {"limit": limit}
        if status:
            params["status"] = status
        with httpx.Client(base_url=self.training_url, follow_redirects=True) as client:
            response = client.get(
                "/v1/datasets/",
                params=params,
                headers=self._auth_headers(),
                timeout=30.0,
            )
            response.raise_for_status()
        return validator.DatasetList.model_validate(response.json())

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def delete(self, dataset_id: str) -> validator.DatasetDeleted:
        with httpx.Client(base_url=self.training_url, follow_redirects=True) as client:
            response = client.delete(
                f"/v1/datasets/{dataset_id}",
                headers=self._auth_headers(),
                timeout=30.0,
            )
            response.raise_for_status()
        return validator.DatasetDeleted.model_validate(response.json())
