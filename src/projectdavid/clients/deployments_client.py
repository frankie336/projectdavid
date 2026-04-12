import os
from typing import Dict, Optional

import httpx
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.deployment_schemas import (
    ActivateBaseModelRequest,
    ActivateFineTunedModelRequest,
    DeactivateAllResponse,
    DeploymentActivationResponse,
    DeploymentDeactivationResponse,
    DeploymentListResponse,
    DeploymentUpdateRequest,
)

from projectdavid.clients.base_client import BaseAPIClient

logging_utility = UtilsInterface.LoggingUtility()


class DeploymentsClient(BaseAPIClient):
    """
    Client for inference deployment lifecycle management.

    Handles activation, updating, and deactivation of base models and
    fine-tuned models on the Project David sovereign AI cluster.

    Activation is asynchronous — the server creates a pending InferenceDeployment
    record and returns immediately. The InferenceReconciler (running inside
    inference_worker) picks it up on its next poll cycle and deploys the
    corresponding Ray Serve application.

    All operations require admin privileges.

    Usage::

        client = Entity(api_key="...")

        # Activate a base model with custom hyperparams
        result = client.deployments.activate_base(
            base_model_id="OpenGVLab/InternVL2-4B",
            gpu_memory_utilization=0.95,
            max_model_len=8192,
            limit_mm_per_prompt={"image": 2},
        )

        # Activate a fine-tuned model
        result = client.deployments.activate_fine_tuned(
            model_id="ftm_G05BERHAEvSRr2KTyUqWIJ",
            gpu_memory_utilization=0.90,
        )

        # Patch a live deployment without reactivating
        client.deployments.update(
            deployment_id="dep_abc123",
            max_model_len=4096,
            enforce_eager=True,
        )

        # List active deployments
        deployments = client.deployments.list()

        # Deactivate a base model
        client.deployments.deactivate_base("OpenGVLab/InternVL2-4B")

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
        # --- vLLM engine hyperparam overrides ---
        # All optional. None = server falls back to VLLM_DEFAULT_* env vars
        # or built-in safe defaults in inference_worker.py.
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        quantization: Optional[str] = None,
        dtype: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
    ) -> DeploymentActivationResponse:
        """
        Schedule a base model (no LoRA adapter) for inference deployment.

        Accepts either a ``bm_...`` prefixed catalog ID or a raw HuggingFace
        model path. The server resolves HF paths to catalog IDs automatically.

        All vLLM hyperparam args are optional. Omit them to use the node-level
        env var defaults. Set them to tune this specific deployment without
        touching compose files or rebuilding images.

        Args:
            base_model_id:           ``bm_...`` ID or HF path.
            target_node_id:          Optional. Pin to a specific Ray node ID.
            tensor_parallel_size:    Number of GPUs to shard across. Default 1.
            gpu_memory_utilization:  VRAM fraction [0.10–0.95]. None = env default.
            max_model_len:           Max sequence length in tokens. None = env default.
            max_num_seqs:            Max concurrent sequences. None = vLLM default.
            quantization:            e.g. 'awq_marlin', 'gptq'. None = full precision.
            dtype:                   e.g. 'float16', 'bfloat16'. None = float16.
            enforce_eager:           Disable CUDA graphs. None = False.
            limit_mm_per_prompt:     e.g. {'image': 2}. None = vLLM default.

        Returns:
            DeploymentActivationResponse with deployment ID, node, and serve route.
        """
        logging_utility.info(
            "DeploymentsClient: activating base model: %s", base_model_id
        )
        payload = ActivateBaseModelRequest(
            base_model_id=base_model_id,
            target_node_id=target_node_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            quantization=quantization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
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
        # --- vLLM engine hyperparam overrides ---
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        quantization: Optional[str] = None,
        dtype: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
    ) -> DeploymentActivationResponse:
        """
        Schedule a fine-tuned model (base + LoRA adapter) for inference deployment.

        All vLLM hyperparam args are optional. Omit them to use the node-level
        env var defaults.

        Args:
            model_id:                ``ftm_...`` prefixed ID of the fine-tuned model.
            target_node_id:          Optional. Pin to a specific Ray node ID.
            tensor_parallel_size:    Number of GPUs to shard across. Default 1.
            gpu_memory_utilization:  VRAM fraction [0.10–0.95]. None = env default.
            max_model_len:           Max sequence length in tokens. None = env default.
            max_num_seqs:            Max concurrent sequences. None = vLLM default.
            quantization:            e.g. 'awq_marlin', 'gptq'. None = full precision.
            dtype:                   e.g. 'float16', 'bfloat16'. None = float16.
            enforce_eager:           Disable CUDA graphs. None = False.
            limit_mm_per_prompt:     e.g. {'image': 2}. None = vLLM default.

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
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            quantization=quantization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
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
    # Update (partial patch)
    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self,
        deployment_id: str,
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        quantization: Optional[str] = None,
        dtype: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        tensor_parallel_size: Optional[int] = None,
    ) -> dict:
        """
        Partially update the vLLM engine hyperparams for a live deployment.

        Only fields explicitly provided are sent to the server — omitted fields
        retain their current DB values. Changes are picked up by the
        InferenceReconciler on its next poll cycle.

        Use this to tune a running deployment without reactivating it:

            client.deployments.update(
                deployment_id="dep_abc123",
                max_model_len=8192,
                gpu_memory_utilization=0.95,
            )

        Args:
            deployment_id:           ``dep_...`` ID of the deployment to patch.
            gpu_memory_utilization:  New VRAM fraction [0.10–0.95].
            max_model_len:           New max sequence length in tokens.
            max_num_seqs:            New max concurrent sequences.
            quantization:            New quantization scheme.
            dtype:                   New compute dtype.
            enforce_eager:           Enable/disable CUDA graphs.
            limit_mm_per_prompt:     New per-modality token cap.
            tensor_parallel_size:    New GPU shard count.

        Returns:
            dict with status, deployment_id, and updated_fields.
        """
        logging_utility.info(
            "DeploymentsClient: patching deployment: %s", deployment_id
        )

        # Build payload — only include fields the caller explicitly passed
        payload = DeploymentUpdateRequest(
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            quantization=quantization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            tensor_parallel_size=tensor_parallel_size,
        ).model_dump(exclude_unset=True)

        try:
            response = self.client.patch(
                f"{self.training_url}/v1/deployments/{deployment_id}",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d patching deployment %s: %s",
                e.response.status_code,
                deployment_id,
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
