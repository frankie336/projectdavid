import os
from typing import Optional

import httpx
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.deployment_schemas import (
    ActivateBaseModelRequest,
    ActivateFineTunedModelRequest,
    DeactivateAllResponse,
    DeploymentActivationResponse,
    DeploymentDeactivationResponse,
    DeploymentListResponse,
)

from projectdavid.clients.base_client import BaseAPIClient

logging_utility = UtilsInterface.LoggingUtility()


class DeploymentsClient(BaseAPIClient):
    """
    Client for inference deployment lifecycle management.

    Handles activation and deactivation of base models and fine-tuned models
    on the Project David sovereign AI cluster.

    Activation is asynchronous — the server creates a pending InferenceDeployment
    record and returns immediately. The InferenceReconciler (running inside
    inference_worker) picks it up on its next poll cycle and deploys the
    corresponding Ray Serve application.

    All activation and deactivation operations require admin privileges.

    Usage::

        client = Entity(api_key="...")

        # Activate a base model (accepts bm_... ID or HF path)
        result = client.deployments.activate_base(
            base_model_id="unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"
        )

        # Activate a fine-tuned model
        result = client.deployments.activate_fine_tuned(
            model_id="ftm_G05BERHAEvSRr2KTyUqWIJ"
        )

        # List active deployments
        deployments = client.deployments.list()

        # Deactivate a base model
        client.deployments.deactivate_base("bm_KZcYp7GJaD4M58gBSlTlsj")

        # Full cluster reset
        client.deployments.deactivate_all()
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
    # Activation
    # ──────────────────────────────────────────────────────────────────────────

    def activate_base(
        self,
        base_model_id: str,
        target_node_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ) -> DeploymentActivationResponse:
        """
        Schedule a base model (no LoRA adapter) for inference deployment.

        Accepts either a ``bm_...`` prefixed catalog ID or a raw HuggingFace
        model path. The server resolves HF paths to catalog IDs automatically.

        Args:
            base_model_id:       ``bm_...`` ID or HF path, e.g.
                                 ``'unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit'``.
            target_node_id:      Optional. Pin to a specific Ray node ID.
                                 If omitted, the server selects the best available node.
            tensor_parallel_size: Number of GPUs to shard the model across. Default 1.

        Returns:
            DeploymentActivationResponse with deployment ID, node, and serve route.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses.
            ValueError: If the model is not registered in the catalog.
        """
        logging_utility.info(
            "DeploymentsClient: activating base model: %s", base_model_id
        )
        payload = ActivateBaseModelRequest(
            base_model_id=base_model_id,
            target_node_id=target_node_id,
            tensor_parallel_size=tensor_parallel_size,
        ).model_dump(exclude_none=True)

        try:
            response = self.client.post(
                f"{self.training_url}/v1/deployments/base",
                json=payload,
            )
            response.raise_for_status()
            return DeploymentActivationResponse.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d activating base model %s: %s",
                e.response.status_code,
                base_model_id,
                e.response.text,
            )
            raise

    def activate_fine_tuned(
        self,
        model_id: str,
        target_node_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ) -> DeploymentActivationResponse:
        """
        Schedule a fine-tuned model (base + LoRA adapter) for inference deployment.

        Args:
            model_id:            ``ftm_...`` prefixed ID of the fine-tuned model.
            target_node_id:      Optional. Pin to a specific Ray node ID.
            tensor_parallel_size: Number of GPUs to shard the model across. Default 1.

        Returns:
            DeploymentActivationResponse with deployment ID, node, and serve route.
        """
        logging_utility.info(
            "DeploymentsClient: activating fine-tuned model: %s", model_id
        )
        payload = ActivateFineTunedModelRequest(
            model_id=model_id,
            target_node_id=target_node_id,
            tensor_parallel_size=tensor_parallel_size,
        ).model_dump(exclude_none=True)

        try:
            response = self.client.post(
                f"{self.training_url}/v1/deployments/fine-tuned",
                json=payload,
            )
            response.raise_for_status()
            return DeploymentActivationResponse.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d activating fine-tuned model %s: %s",
                e.response.status_code,
                model_id,
                e.response.text,
            )
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Listing
    # ──────────────────────────────────────────────────────────────────────────

    def list(self) -> DeploymentListResponse:
        """
        Return all active InferenceDeployment records.

        Returns:
            DeploymentListResponse containing items and total count.
        """
        logging_utility.info("DeploymentsClient: listing active deployments")
        try:
            response = self.client.get(
                f"{self.training_url}/v1/deployments/",
            )
            response.raise_for_status()
            return DeploymentListResponse.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d listing deployments: %s",
                e.response.status_code,
                e.response.text,
            )
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Deactivation
    # ──────────────────────────────────────────────────────────────────────────

    def deactivate_base(
        self,
        base_model_id: str,
    ) -> DeploymentDeactivationResponse:
        """
        Surgically deactivate a single base model deployment.

        Accepts either a ``bm_...`` prefixed catalog ID or a raw HuggingFace
        model path.

        Args:
            base_model_id: ``bm_...`` ID or HF path of the base model to deactivate.

        Returns:
            DeploymentDeactivationResponse confirming deactivation.
        """
        logging_utility.info(
            "DeploymentsClient: deactivating base model: %s", base_model_id
        )
        try:

            response = self.client.delete(
                f"{self.training_url}/v1/deployments/base/{base_model_id}",
            )

            response.raise_for_status()
            return DeploymentDeactivationResponse.model_validate(response.json())

        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d deactivating base model %s: %s",
                e.response.status_code,
                base_model_id,
                e.response.text,
            )
            raise

    def deactivate_fine_tuned(
        self,
        model_id: str,
    ) -> DeploymentDeactivationResponse:
        """
        Surgically deactivate a single fine-tuned model deployment.

        Args:
            model_id: ``ftm_...`` prefixed ID of the fine-tuned model to deactivate.

        Returns:
            DeploymentDeactivationResponse confirming deactivation.
        """
        logging_utility.info(
            "DeploymentsClient: deactivating fine-tuned model: %s", model_id
        )
        try:
            response = self.client.delete(
                f"{self.training_url}/v1/deployments/fine-tuned/{model_id}",
            )
            response.raise_for_status()
            return DeploymentDeactivationResponse.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d deactivating fine-tuned model %s: %s",
                e.response.status_code,
                model_id,
                e.response.text,
            )
            raise

    def deactivate_all(self) -> DeactivateAllResponse:
        """
        Full cluster reset — deactivate all active deployments.

        Removes all InferenceDeployment records. The InferenceReconciler
        tears down all Ray Serve applications on its next poll cycle,
        releasing all GPU reservations back to the cluster.

        Returns:
            DeactivateAllResponse confirming the reset.
        """
        logging_utility.warning(
            "DeploymentsClient: requesting full cluster reset (deactivate-all)"
        )
        try:
            response = self.client.delete(
                f"{self.training_url}/v1/deployments/",
            )
            response.raise_for_status()
            return DeactivateAllResponse.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d on deactivate-all: %s",
                e.response.status_code,
                e.response.text,
            )
            raise
