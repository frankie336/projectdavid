import os
from typing import Any, Dict, Optional

from projectdavid_common import UtilsInterface, ValidationInterface

from projectdavid.clients.base_client import BaseAPIClient

logging_utility = UtilsInterface.LoggingUtility()
validator = ValidationInterface()


class ModelsClient(BaseAPIClient):
    """
    Client for managing the Fine-Tuned Model Registry and Cluster Deployments.

    Supports:
      - Model CRUD (Metadata)
      - Fine-Tuned Model Activation (LoRA)
      - Base Model Activation (Standard Backbone)
      - Surgical and Global Deactivation
      - Hardware Node Pinning
    """

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

    # ──────────────────────────────────────────────────────────────────────────
    # REGISTRY MANAGEMENT (Metadata CRUD)
    # ──────────────────────────────────────────────────────────────────────────

    def list(self, limit: int = 50, offset: int = 0) -> validator.FineTunedModelList:
        """List all fine-tuned models for the current user."""
        response = self.client.get(
            f"{self.training_url}/v1/fine-tuned-models/",
            params={"limit": limit, "offset": offset},
        )
        response.raise_for_status()
        return validator.FineTunedModelList.model_validate(response.json())

    def retrieve(self, model_id: str) -> validator.FineTunedModelRead:
        """Fetch metadata for a specific fine-tuned model."""
        response = self.client.get(
            f"{self.training_url}/v1/fine-tuned-models/{model_id}"
        )
        response.raise_for_status()
        return validator.FineTunedModelRead.model_validate(response.json())

    def delete(self, model_id: str) -> validator.FineTunedModelDeleted:
        """Soft-delete a fine-tuned model from the registry."""
        response = self.client.delete(
            f"{self.training_url}/v1/fine-tuned-models/{model_id}"
        )
        response.raise_for_status()
        return validator.FineTunedModelDeleted.model_validate(response.json())

    # ──────────────────────────────────────────────────────────────────────────
    # FINE-TUNED MODEL LIFECYCLE (LoRA Adapters)
    # ──────────────────────────────────────────────────────────────────────────

    def activate(
        self, model_id: str, node_id: Optional[str] = None
    ) -> validator.ActivateModelResponse:
        """
        Promote a fine-tuned model to 'Active' status on the cluster.

        Args:
            model_id: The ftm_... ID of the model.
            node_id: Optional. The ID of a specific GPU node to pin this model to.
        """
        logging_utility.info("🚀 Requesting activation for model: %s", model_id)
        params = {"node_id": node_id} if node_id else {}

        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/{model_id}/activate",
            params=params,
        )
        response.raise_for_status()
        return validator.ActivateModelResponse.model_validate(response.json())

    def deactivate(self, model_id: str) -> dict:
        """
        Surgically shut down a specific fine-tuned model deployment.
        This releases the VRAM on the hosting node.
        """
        logging_utility.info("🛑 Deactivating fine-tuned model: %s", model_id)
        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/{model_id}/deactivate"
        )
        response.raise_for_status()
        return response.json()

    # ──────────────────────────────────────────────────────────────────────────
    # BASE MODEL LIFECYCLE (Factory Backbones)
    # ──────────────────────────────────────────────────────────────────────────
    def activate_base(self, base_model_id: str, node_id: Optional[str] = None) -> dict:
        """
        Deploy a standard backbone model (no LoRA) to the mesh.

        Args:
            base_model_id: e.g. 'unsloth/Llama-3.2-1B-Instruct-bnb-4bit'
            node_id: Optional. Target a specific machine in the mesh.
        """
        logging_utility.info("🚀 Requesting base model deployment: %s", base_model_id)
        params = {"node_id": node_id} if node_id else {}

        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/base/{base_model_id}/activate",
            params=params,
        )
        response.raise_for_status()
        return response.json()

    def deactivate_base(self, base_model_id: str) -> dict:
        """Shut down a specific standard backbone deployment."""
        logging_utility.info("🛑 Deactivating base model: %s", base_model_id)
        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/base/{base_model_id}/deactivate"
        )
        response.raise_for_status()
        return response.json()

    # ──────────────────────────────────────────────────────────────────────────
    # CLUSTER MAINTENANCE
    # ──────────────────────────────────────────────────────────────────────────

    def deactivate_all(self) -> dict:
        """
        Emergency Stop: Deactivate any currently active model (Base or LoRA).
        Reverts the cluster to an idle/factory state.
        """
        logging_utility.warning(
            "⚠️ Requesting global cluster reset (deactivate-all)..."
        )
        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/deactivate-all"
        )
        response.raise_for_status()
        return response.json()
