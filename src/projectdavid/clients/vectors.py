import asyncio
import os
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import BaseModel, Field

from projectdavid.clients.file_processor import FileProcessor
from projectdavid.clients.vector_store_manager import VectorStoreManager

load_dotenv()
log = UtilsInterface.LoggingUtility()


class VectorStoreClientError(Exception):
    """Custom exception for VectorStoreClient errors."""

    pass


class VectorStoreFileUpdateStatusInput(BaseModel):
    status: ValidationInterface.StatusEnum = Field(
        ..., description="The new status for the file record."
    )
    error_message: Optional[str] = Field(
        None, description="Error message if status is 'failed'."
    )


class VectorStoreClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_store_host: Optional[str] = "localhost",
    ):
        self.base_url = (base_url or os.getenv("BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY")

        if not self.base_url:
            raise VectorStoreClientError("BASE_URL is required.")

        # single headers dict
        self._base_headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            self._base_headers["X-API-Key"] = self.api_key
            log.info("API Key provided and added to headers.")
        else:
            log.warning("No API Key provided; requests may fail.")

        # only keep sync client for direct sync calls
        self._sync_api_client = httpx.Client(
            base_url=self.base_url, headers=self._base_headers, timeout=30.0
        )

        # other services
        self.vector_store_host = vector_store_host
        self.vector_manager = VectorStoreManager(vector_store_host=vector_store_host)
        self.identifier_service = UtilsInterface.IdentifierService()
        self.file_processor = FileProcessor()

        log.info("VectorStoreClient initialized with base_url: %s", self.base_url)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        """Async cleanup: close the sync client on a thread."""
        await asyncio.to_thread(self._sync_api_client.close)

    def close(self):
        """Sync cleanup helper that works inside/outside loops."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                warnings.warn(
                    "close() called inside running loop – use aclose() instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                try:
                    self._sync_api_client.close()
                except Exception:
                    pass
                return
        except RuntimeError:
            pass  # no loop running
        asyncio.run(self.aclose())

    async def _internal_parse_response(self, response: httpx.Response) -> Any:
        try:
            response.raise_for_status()
            return None if response.status_code == 204 else response.json()
        except httpx.HTTPStatusError as e:
            log.error(
                "API request failed: %d – %s", e.response.status_code, e.response.text
            )
            raise VectorStoreClientError(
                f"API Error: {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            log.error("Failed to parse API response: %s", e)
            raise VectorStoreClientError(
                f"Invalid response from API: {response.text}"
            ) from e

    async def _internal_request_with_retries(
        self, method: str, url: str, **kwargs
    ) -> Any:
        """
        Creates a fresh AsyncClient for each attempt to avoid
        event-loop-closed errors, with simple exponential backoff.
        """
        retries = 3
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=self._base_headers,
                    timeout=30.0,
                ) as client:
                    resp = await client.request(method, url, **kwargs)
                    return await self._internal_parse_response(resp)

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
            ) as e:
                last_exc = e
                should_retry = isinstance(
                    e, (httpx.TimeoutException, httpx.NetworkError)
                ) or (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code >= 500
                )
                if should_retry and attempt < retries:
                    backoff = 2 ** (attempt - 1)
                    log.warning(
                        "Retry %d/%d %s %s in %ds — %s",
                        attempt,
                        retries,
                        method,
                        url,
                        backoff,
                        e,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise VectorStoreClientError(str(e)) from e

            except Exception as e:
                raise VectorStoreClientError(f"Unexpected error: {e}") from e

        raise VectorStoreClientError("Request failed after retries") from last_exc

    # ——— Internal async implementations ———

    async def _internal_create_vector_store_async(
        self,
        name: str,
        user_id: str,
        vector_size: int,
        distance_metric: str,
        config: Optional[Dict[str, Any]],
    ) -> ValidationInterface.VectorStoreRead:
        shared_id = self.identifier_service.generate_vector_id()
        log.info("Attempting to create Qdrant collection '%s'", shared_id)
        try:
            self.vector_manager.create_store(
                collection_name=shared_id,
                vector_size=vector_size,
                distance=distance_metric.upper(),
            )
            log.info("Qdrant collection '%s' created", shared_id)
        except Exception as e:
            log.error("Qdrant create failed for '%s': %s", shared_id, e)
            raise VectorStoreClientError(f"Backend create failed: {e}") from e

        payload = {
            "shared_id": shared_id,
            "name": name,
            "user_id": user_id,
            "vector_size": vector_size,
            "distance_metric": distance_metric.upper(),
            "config": config or {},
        }
        log.info("Registering vector store '%s' via API", name)
        resp = await self._internal_request_with_retries(
            "POST", "/v1/vector-stores", json=payload
        )
        return ValidationInterface.VectorStoreRead.model_validate(resp)

    async def _internal_add_file_to_vector_store_async(
        self,
        vector_store_id: str,
        file_path: Path,
        user_metadata: Optional[Dict[str, Any]],
    ) -> ValidationInterface.VectorStoreFileRead:
        log.info("Processing file %s for store %s", file_path, vector_store_id)
        processed = await self.file_processor.process_file(file_path)
        texts, vectors = processed["chunks"], processed["vectors"]
        if not texts or not vectors:
            raise VectorStoreClientError(f"No content in {file_path.name}")

        base_md = user_metadata or {}
        base_md.update({"source": str(file_path), "file_name": file_path.name})
        chunk_md = [{**base_md, "chunk_index": i} for i in range(len(texts))]

        log.info("Uploading %d chunks to Qdrant", len(texts))
        try:
            self.vector_manager.add_to_store(
                store_name=vector_store_id,
                texts=texts,
                vectors=vectors,
                metadata=chunk_md,
            )
        except Exception as e:
            log.error("Qdrant upload failed: %s", e, exc_info=True)
            raise VectorStoreClientError(f"Upload failed: {e}") from e

        file_id = f"vsf_{uuid.uuid4()}"
        payload = {
            "file_id": file_id,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "status": "completed",
            "meta_data": user_metadata or {},
        }
        log.info("Registering file %s via API", file_path.name)
        resp = await self._internal_request_with_retries(
            "POST", f"/v1/vector-stores/{vector_store_id}/files", json=payload
        )
        return ValidationInterface.VectorStoreFileRead.model_validate(resp)

    async def _internal_search_vector_store_async(
        self, vector_store_id: str, query_text: str, top_k: int, filters: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        store = self.retrieve_vector_store_sync(vector_store_id)
        vec = self.file_processor.embedding_model.encode(query_text).tolist()
        return self.vector_manager.query_store(
            store_name=store.collection_name,
            query_vector=vec,
            top_k=top_k,
            filters=filters,
        )

    async def _internal_delete_vector_store_async(
        self, vector_store_id: str, permanent: bool
    ) -> Dict[str, Any]:
        qres = self.vector_manager.delete_store(vector_store_id)
        await self._internal_request_with_retries(
            "DELETE",
            f"/v1/vector-stores/{vector_store_id}",
            params={"permanent": permanent},
        )
        return {
            "vector_store_id": vector_store_id,
            "status": "deleted",
            "permanent": permanent,
            "qdrant_result": qres,
        }

    async def _internal_delete_file_from_vector_store_async(
        self, vector_store_id: str, file_path: str
    ) -> Dict[str, Any]:
        fres = self.vector_manager.delete_file_from_store(vector_store_id, file_path)
        await self._internal_request_with_retries(
            "DELETE",
            f"/v1/vector-stores/{vector_store_id}/files",
            params={"file_path": file_path},
        )
        return {
            "vector_store_id": vector_store_id,
            "file_path": file_path,
            "status": "deleted",
            "qdrant_result": fres,
        }

    async def _internal_list_store_files_async(
        self, vector_store_id: str
    ) -> List[ValidationInterface.VectorStoreFileRead]:
        resp = await self._internal_request_with_retries(
            "GET", f"/v1/vector-stores/{vector_store_id}/files"
        )
        return [
            ValidationInterface.VectorStoreFileRead.model_validate(item)
            for item in resp
        ]

    async def _internal_update_file_status_async(
        self,
        vector_store_id: str,
        file_id: str,
        status: ValidationInterface.StatusEnum,
        error_message: Optional[str] = None,
    ) -> ValidationInterface.VectorStoreFileRead:
        payload = VectorStoreFileUpdateStatusInput(
            status=status, error_message=error_message
        ).model_dump(exclude_none=True)
        resp = await self._internal_request_with_retries(
            "PATCH",
            f"/v1/vector-stores/{vector_store_id}/files/{file_id}",
            json=payload,
        )
        return ValidationInterface.VectorStoreFileRead.model_validate(resp)

    async def _internal_get_assistant_vs_async(
        self, assistant_id: str
    ) -> List[ValidationInterface.VectorStoreRead]:
        resp = await self._internal_request_with_retries(
            "GET", f"/v1/assistants/{assistant_id}/vector-stores"
        )
        return [
            ValidationInterface.VectorStoreRead.model_validate(item) for item in resp
        ]

    async def _internal_get_user_vs_async(
        self, user_id: str
    ) -> List[ValidationInterface.VectorStoreRead]:
        resp = await self._internal_request_with_retries(
            "GET", f"/v1/users/{user_id}/vector-stores"
        )
        return [
            ValidationInterface.VectorStoreRead.model_validate(item) for item in resp
        ]

    async def _internal_retrieve_vs_async(
        self, vector_store_id: str
    ) -> ValidationInterface.VectorStoreRead:
        resp = await self._internal_request_with_retries(
            "GET", f"/v1/vector-stores/{vector_store_id}"
        )
        return ValidationInterface.VectorStoreRead.model_validate(resp)

    async def _internal_attach_vector_store_to_assistant_async(
        self, vector_store_id: str, assistant_id: str
    ) -> bool:
        await self._internal_request_with_retries(
            "POST",
            f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/attach",
        )
        return True  # 200 with {"success": true}

    async def _internal_detach_vector_store_from_assistant_async(
        self, vector_store_id: str, assistant_id: str
    ) -> bool:
        await self._internal_request_with_retries(
            "DELETE",
            f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/detach",
        )
        return True

    # ——— Public sync methods ———

    def _run_sync(self, coro):
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise VectorStoreClientError(
                    "Cannot call sync method inside running loop."
                )
        except RuntimeError:
            pass
        return asyncio.run(coro)

    def create_vector_store(
        self,
        name: str,
        user_id: str,
        vector_size: int = 384,
        distance_metric: str = "Cosine",
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreRead:
        return self._run_sync(
            self._internal_create_vector_store_async(
                name, user_id, vector_size, distance_metric, config
            )
        )

    def add_file_to_vector_store(
        self,
        vector_store_id: str,
        file_path: Union[str, Path],
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreFileRead:
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        return self._run_sync(
            self._internal_add_file_to_vector_store_async(
                vector_store_id, p, user_metadata
            )
        )

    def search_vector_store(
        self,
        vector_store_id: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        return self._run_sync(
            self._internal_search_vector_store_async(
                vector_store_id, query_text, top_k, filters
            )
        )

    def delete_vector_store(
        self, vector_store_id: str, permanent: bool = False
    ) -> Dict[str, Any]:
        return self._run_sync(
            self._internal_delete_vector_store_async(vector_store_id, permanent)
        )

    def delete_file_from_vector_store(
        self, vector_store_id: str, file_path: str
    ) -> Dict[str, Any]:
        return self._run_sync(
            self._internal_delete_file_from_vector_store_async(
                vector_store_id, file_path
            )
        )

    def list_store_files(
        self, vector_store_id: str
    ) -> List[ValidationInterface.VectorStoreFileRead]:
        return self._run_sync(self._internal_list_store_files_async(vector_store_id))

    def update_vector_store_file_status(
        self,
        vector_store_id: str,
        file_id: str,
        status: ValidationInterface.StatusEnum,
        error_message: Optional[str] = None,
    ) -> ValidationInterface.VectorStoreFileRead:
        return self._run_sync(
            self._internal_update_file_status_async(
                vector_store_id, file_id, status, error_message
            )
        )

    def get_vector_stores_for_assistant(
        self, assistant_id: str
    ) -> List[ValidationInterface.VectorStoreRead]:
        return self._run_sync(self._internal_get_assistant_vs_async(assistant_id))

    def get_stores_by_user(
        self, user_id: str
    ) -> List[ValidationInterface.VectorStoreRead]:
        return self._run_sync(self._internal_get_user_vs_async(user_id))

    def retrieve_vector_store(
        self, vector_store_id: str
    ) -> ValidationInterface.VectorStoreRead:
        return self._run_sync(self._internal_retrieve_vs_async(vector_store_id))

    def attach_vector_store_to_assistant(
        self, vector_store_id: str, assistant_id: str
    ) -> bool:
        """
        Link a vector‑store to an assistant (idempotent, raises on 4xx/5xx).
        """
        return self._run_sync(
            self._internal_attach_vector_store_to_assistant_async(
                vector_store_id, assistant_id
            )
        )

    def detach_vector_store_from_assistant(
        self, vector_store_id: str, assistant_id: str
    ) -> bool:
        """
        Unlink a vector‑store from an assistant (idempotent).
        """
        return self._run_sync(
            self._internal_detach_vector_store_from_assistant_async(
                vector_store_id, assistant_id
            )
        )

    def retrieve_vector_store_sync(
        self, vector_store_id: str
    ) -> ValidationInterface.VectorStoreRead:
        log.info("Retrieving vector store %s via sync client", vector_store_id)
        resp = self._sync_api_client.get(f"/v1/vector-stores/{vector_store_id}")
        resp.raise_for_status()
        return ValidationInterface.VectorStoreRead.model_validate(resp.json())
