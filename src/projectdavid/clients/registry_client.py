# projectdavid/clients/registry_client.py

from typing import Optional

import httpx
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

from projectdavid.clients.base_client import BaseAPIClient

logging_utility = UtilsInterface.LoggingUtility()
validator = ValidationInterface()


class RegistryClient(BaseAPIClient):
    """
    Client for the Base Model Registry API.

    Supports:
      - Registering HuggingFace base models
      - Listing the model catalog
      - Retrieving a model by bm_... ID or HF path
      - Deregistering a model (admin only)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        training_url: Optional[str] = None,
    ):
        super().__init__(base_url=base_url, api_key=api_key)
        import os

        resolved_url = (
            training_url or os.getenv("TRAINING_BASE_URL") or "http://localhost:9001"
        )
        self.training_url = resolved_url.rstrip("/")

        logging_utility.info(
            "RegistryClient initialized with training_url: %s", self.training_url
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        hf_model_id: str,
        name: str,
        family: Optional[str] = None,
        parameter_count: Optional[str] = None,
        is_multimodal: bool = False,
    ) -> validator.BaseModelRead:
        """
        Register a HuggingFace base model in the catalog.

        Idempotent — re-registering the same HF path returns the
        existing record without error.

        Args:
            hf_model_id:     HuggingFace model path,
                             e.g. 'unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit'
            name:            Human-readable display name.
            family:          Model family, e.g. 'qwen', 'llama', 'mistral'.
            parameter_count: Parameter count string, e.g. '1.5B', '7B'.
            is_multimodal:   True if the model accepts image inputs.

        Returns:
            BaseModelRead — the registered (or existing) model record.
        """
        logging_utility.info("RegistryClient: registering base model: %s", hf_model_id)
        payload = validator.BaseModelRegisterRequest(
            hf_model_id=hf_model_id,
            name=name,
            family=family,
            parameter_count=parameter_count,
            is_multimodal=is_multimodal,
        ).model_dump(exclude_none=True)

        try:
            response = self.client.post(
                f"{self.training_url}/v1/registry/base-models",
                json=payload,
            )
            response.raise_for_status()
            return validator.BaseModelRead.model_validate(response.json())

        except ValidationError as e:
            logging_utility.error("Validation error registering model: %s", e.json())
            raise ValueError(f"Validation error: {e}") from e
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d while registering model: %s",
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception:
            logging_utility.exception("Unexpected error registering base model")
            raise

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> validator.BaseModelList:
        """
        Return a paginated list of all registered base models.

        Args:
            limit:  Page size (1–200).
            offset: Pagination offset.

        Returns:
            BaseModelList — items, total, limit, offset.
        """
        logging_utility.info(
            "RegistryClient: listing base models (limit=%d offset=%d)", limit, offset
        )
        try:
            response = self.client.get(
                f"{self.training_url}/v1/registry/base-models",
                params={"limit": limit, "offset": offset},
            )
            response.raise_for_status()
            return validator.BaseModelList.model_validate(response.json())

        except ValidationError as e:
            logging_utility.error("Validation error listing models: %s", e.json())
            raise ValueError(f"Validation error: {e}") from e
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d while listing models: %s",
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception:
            logging_utility.exception("Unexpected error listing base models")
            raise

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, model_ref: str) -> validator.BaseModelRead:
        """
        Fetch a base model by its bm_... prefixed ID or HF model path.

        Args:
            model_ref: Either a bm_... ID or an HF path such as
                       'unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit'.

        Returns:
            BaseModelRead — the matched model record.
        """
        logging_utility.info("RegistryClient: retrieving base model: %s", model_ref)
        try:
            response = self.client.get(
                f"{self.training_url}/v1/registry/base-models/{model_ref}",
            )
            response.raise_for_status()
            return validator.BaseModelRead.model_validate(response.json())

        except ValidationError as e:
            logging_utility.error("Validation error retrieving model: %s", e.json())
            raise ValueError(f"Validation error: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logging_utility.warning("Base model not found: %s", model_ref)
            else:
                logging_utility.error(
                    "HTTP %d while retrieving model: %s",
                    e.response.status_code,
                    e.response.text,
                )
            raise
        except Exception:
            logging_utility.exception("Unexpected error retrieving base model")
            raise

    # ------------------------------------------------------------------
    # Deregistration
    # ------------------------------------------------------------------

    def deregister(self, model_id: str) -> validator.BaseModelDeleted:
        """
        Remove a base model from the catalog by its bm_... prefixed ID.

        Admin only — the server will return 403 if the caller does not
        have admin privileges.

        Args:
            model_id: The bm_... prefixed ID of the model to remove.

        Returns:
            BaseModelDeleted — confirmation payload with status and model_id.
        """
        logging_utility.warning(
            "RegistryClient: deregistering base model: %s", model_id
        )
        try:
            response = self.client.delete(
                f"{self.training_url}/v1/registry/base-models/{model_id}",
            )
            response.raise_for_status()
            return validator.BaseModelDeleted.model_validate(response.json())

        except ValidationError as e:
            logging_utility.error("Validation error on deregister: %s", e.json())
            raise ValueError(f"Validation error: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logging_utility.error(
                    "Deregister rejected — admin privileges required."
                )
            elif e.response.status_code == 404:
                logging_utility.warning(
                    "Base model not found for deregistration: %s", model_id
                )
            else:
                logging_utility.error(
                    "HTTP %d while deregistering model: %s",
                    e.response.status_code,
                    e.response.text,
                )
            raise
        except Exception:
            logging_utility.exception("Unexpected error deregistering base model")
            raise
