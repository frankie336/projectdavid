import os
import warnings
from typing import Any, Dict, Optional

from projectdavid_common import UtilsInterface, ValidationInterface

from projectdavid.clients.base_client import BaseAPIClient

logging_utility = UtilsInterface.LoggingUtility()
validator = ValidationInterface()

_DEPRECATION_NOTE = (
    "This method is deprecated and will be removed in v3.0.0. "
    "Use the DeploymentsClient instead: client.deployments.{method}(). "
    "See https://docs.projectdavid.co.uk/docs/migration/v2-to-v3 for details."
)


class ModelsClient(BaseAPIClient):
    """
    Client for managing Fine-Tuned Model metadata and cluster deployments.

    .. deprecated::
        Activation and deactivation methods on this client are deprecated.
        Use ``client.deployments`` instead for all deployment lifecycle operations.
        These methods will be removed in v3.0.0.

    Supports:
      - Model CRUD (list, retrieve, delete)
      - Fine-Tuned Model Activation [DEPRECATED — use client.deployments.activate_fine_tuned()]
      - Base Model Activation       [DEPRECATED — use client.deployments.activate_base()]
      - Deactivation                [DEPRECATED — use client.deployments.deactivate_*()]
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        training_url: Optional[str] = None,
    ):
        super().__init__(base_url=base_url, api_key=api_key)

        resolved_url = (
            training_url
            or os.getenv("TRAINING_BASE_URL")
            or base_url
            or "http://localhost:80"
        )
        self.training_url = resolved_url.rstrip("/")

    # ──────────────────────────────────────────────────────────────────────────
    # REGISTRY MANAGEMENT (Metadata CRUD)
    # These methods are NOT deprecated — fine-tuned model CRUD stays here.
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
    # DEPRECATED — use client.deployments.activate_fine_tuned() instead
    # ──────────────────────────────────────────────────────────────────────────

    def activate(
        self,
        model_id: str,
        node_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ) -> validator.ActivateModelResponse:
        """
        Promote a fine-tuned model to Active status on the cluster.

        .. deprecated::
            Use ``client.deployments.activate_fine_tuned(model_id=...)`` instead.
            This method will be removed in v3.0.0.
        """
        warnings.warn(
            f"ModelsClient.activate() is deprecated. "
            f"{_DEPRECATION_NOTE.format(method='activate_fine_tuned')}",
            DeprecationWarning,
            stacklevel=2,
        )
        logging_utility.warning(
            "⚠️  DEPRECATED: ModelsClient.activate() — use client.deployments.activate_fine_tuned()"
        )
        params: Dict[str, Any] = {"tensor_parallel_size": tensor_parallel_size}
        if node_id:
            params["node_id"] = node_id

        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/{model_id}/activate",
            params=params,
        )
        response.raise_for_status()
        return validator.ActivateModelResponse.model_validate(response.json())

    def deactivate(self, model_id: str) -> dict:
        """
        Surgically shut down a specific fine-tuned model deployment.

        .. deprecated::
            Use ``client.deployments.deactivate_fine_tuned(model_id=...)`` instead.
            This method will be removed in v3.0.0.
        """
        warnings.warn(
            f"ModelsClient.deactivate() is deprecated. "
            f"{_DEPRECATION_NOTE.format(method='deactivate_fine_tuned')}",
            DeprecationWarning,
            stacklevel=2,
        )
        logging_utility.warning(
            "⚠️  DEPRECATED: ModelsClient.deactivate() — use client.deployments.deactivate_fine_tuned()"
        )
        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/{model_id}/deactivate"
        )
        response.raise_for_status()
        return response.json()

    # ──────────────────────────────────────────────────────────────────────────
    # BASE MODEL LIFECYCLE (Factory Backbones)
    # DEPRECATED — use client.deployments.activate_base() instead
    # ──────────────────────────────────────────────────────────────────────────

    def activate_base(
        self,
        base_model_id: str,
        node_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ) -> dict:
        """
        Deploy a standard backbone model (no LoRA) to the mesh.

        .. deprecated::
            Use ``client.deployments.activate_base(base_model_id=...)`` instead.
            This method will be removed in v3.0.0.
        """
        warnings.warn(
            f"ModelsClient.activate_base() is deprecated. "
            f"{_DEPRECATION_NOTE.format(method='activate_base')}",
            DeprecationWarning,
            stacklevel=2,
        )
        logging_utility.warning(
            "⚠️  DEPRECATED: ModelsClient.activate_base() — use client.deployments.activate_base()"
        )
        params: Dict[str, Any] = {"tensor_parallel_size": tensor_parallel_size}
        if node_id:
            params["node_id"] = node_id

        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/base/{base_model_id}/activate",
            params=params,
        )
        response.raise_for_status()
        return response.json()

    def deactivate_base(self, base_model_id: str) -> dict:
        """
        Shut down a specific standard backbone deployment.

        .. deprecated::
            Use ``client.deployments.deactivate_base(base_model_id=...)`` instead.
            This method will be removed in v3.0.0.
        """
        warnings.warn(
            f"ModelsClient.deactivate_base() is deprecated. "
            f"{_DEPRECATION_NOTE.format(method='deactivate_base')}",
            DeprecationWarning,
            stacklevel=2,
        )
        logging_utility.warning(
            "⚠️  DEPRECATED: ModelsClient.deactivate_base() — use client.deployments.deactivate_base()"
        )
        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/base/{base_model_id}/deactivate"
        )
        response.raise_for_status()
        return response.json()

    # ──────────────────────────────────────────────────────────────────────────
    # CLUSTER MAINTENANCE
    # DEPRECATED — use client.deployments.deactivate_all() instead
    # ──────────────────────────────────────────────────────────────────────────

    def deactivate_all(self) -> dict:
        """
        Emergency Stop: Deactivate any currently active model.

        .. deprecated::
            Use ``client.deployments.deactivate_all()`` instead.
            This method will be removed in v3.0.0.
        """
        warnings.warn(
            f"ModelsClient.deactivate_all() is deprecated. "
            f"{_DEPRECATION_NOTE.format(method='deactivate_all')}",
            DeprecationWarning,
            stacklevel=2,
        )
        logging_utility.warning(
            "⚠️  DEPRECATED: ModelsClient.deactivate_all() — use client.deployments.deactivate_all()"
        )
        response = self.client.post(
            f"{self.training_url}/v1/fine-tuned-models/deactivate-all"
        )
        response.raise_for_status()
        return response.json()
