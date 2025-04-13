#!src/projectdavid/clients/vectors.py
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
logging_utility = UtilsInterface.LoggingUtility()


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

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            logging_utility.info("API Key provided and added to headers.")
        else:
            logging_utility.warning("No API Key provided; requests may fail.")

        self._async_api_client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=30.0
        )
        self._sync_api_client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=30.0
        )

        self.vector_store_host = vector_store_host
        self.vector_manager = VectorStoreManager(vector_store_host=vector_store_host)
        self.identifier_service = UtilsInterface.IdentifierService()
        self.file_processor = FileProcessor()

        logging_utility.info(
            "VectorStoreClient initialized with base_url: %s", self.base_url
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    async def aclose(self):
        await self._async_api_client.aclose()
        await asyncio.to_thread(self._sync_api_client.close)

    def close(self):
        """Synchronously closes the underlying HTTP clients."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                warnings.warn(
                    "Calling synchronous close() from within a running event loop is problematic. Use aclose() instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                try:
                    self._sync_api_client.close()
                except Exception:
                    pass
                logging_utility.warning(
                    "Synchronous close called from running loop may not fully close async resources."
                )
                return
        except RuntimeError:
            pass  # No loop running
        try:
            asyncio.run(self.aclose())
        except Exception as e:
            logging_utility.error(f"Error during client closure: {e}", exc_info=False)

    # --- Internal Async Helper Methods (Private) ---
    async def _internal_parse_response(self, response: httpx.Response) -> Any:
        try:
            response.raise_for_status()
            if response.status_code == 204:
                return None
            return response.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "API request failed: Status %d, Response: %s",
                e.response.status_code,
                e.response.text,
            )
            raise VectorStoreClientError(
                f"API Error: {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            logging_utility.error("Failed to parse API response: %s", str(e))
            raise VectorStoreClientError(
                f"Invalid response from API: {response.text}"
            ) from e

    async def _internal_request_with_retries(
        self, method: str, url: str, **kwargs
    ) -> Any:
        retries = 3
        last_exception = None
        for attempt in range(retries):
            try:
                response = await self._async_api_client.request(method, url, **kwargs)
                return await self._internal_parse_response(response)
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
            ) as e:
                last_exception = e
                should_retry = isinstance(
                    e, (httpx.TimeoutException, httpx.NetworkError)
                ) or (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code >= 500
                )
                if should_retry and attempt < retries - 1:
                    wait_time = 2**attempt
                    logging_utility.warning(
                        "Retrying request (attempt %d/%d) to %s %s after %d s. Error: %s",
                        attempt + 1,
                        retries,
                        method,
                        url,
                        wait_time,
                        str(e),
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logging_utility.error(
                        "API Request failed permanently after %d attempts to %s %s. Last Error: %s",
                        attempt + 1,
                        method,
                        url,
                        str(e),
                    )
                    if isinstance(e, httpx.HTTPStatusError):
                        raise VectorStoreClientError(
                            f"API Error: {e.response.status_code} - {e.response.text}"
                        ) from e
                    else:
                        raise VectorStoreClientError(
                            f"API Communication Error: {str(e)}"
                        ) from e
            except Exception as e:
                logging_utility.error(
                    "Unexpected error during API request to %s %s: %s",
                    method,
                    url,
                    str(e),
                )
                raise VectorStoreClientError(
                    f"Unexpected API Client Error: {str(e)}"
                ) from e
        raise VectorStoreClientError(
            "Request failed after retries."
        ) from last_exception

    # --- Internal Async Implementations ---

    async def _internal_create_vector_store_async(
        self,
        name: str,
        user_id: str,
        vector_size: int,
        distance_metric: str,
        config: Optional[Dict[str, Any]],
    ) -> ValidationInterface.VectorStoreRead:
        shared_id = self.identifier_service.generate_vector_id()
        backend_collection_name = shared_id
        logging_utility.info(
            "Attempting to create Qdrant collection '%s'", backend_collection_name
        )
        try:
            # Call assumes manager's create_store expects collection_name
            _ = self.vector_manager.create_store(
                collection_name=backend_collection_name,
                vector_size=vector_size,
                distance=distance_metric.upper(),
            )
            logging_utility.info(
                "Successfully created Qdrant collection '%s'", backend_collection_name
            )
        except Exception as e:
            logging_utility.error(
                "Qdrant collection creation failed for '%s': %s",
                backend_collection_name,
                str(e),
            )
            raise VectorStoreClientError(
                f"Failed to create vector store backend: {str(e)}"
            ) from e

        logging_utility.info(
            "Registering vector store '%s' (ID: %s) via API", name, shared_id
        )
        # Payload matches VectorStoreCreateWithSharedId model used in API
        db_payload = {
            "shared_id": shared_id,
            "name": name,
            "user_id": user_id,
            "vector_size": vector_size,
            "distance_metric": distance_metric.upper(),
            "config": config or {},
        }
        try:
            response_data = await self._internal_request_with_retries(
                "POST", "/v1/vector-stores", json=db_payload
            )
            logging_utility.info(
                "Successfully registered vector store '%s' via API", name
            )
            return ValidationInterface.VectorStoreRead.model_validate(response_data)
        except Exception as api_error:
            logging_utility.error(
                "API registration failed for store '%s' (ID: %s). Rolling back Qdrant collection. Error: %s",
                name,
                shared_id,
                str(api_error),
            )
            try:
                self.vector_manager.delete_store(backend_collection_name)
                logging_utility.info(
                    "Rolled back Qdrant collection '%s'", backend_collection_name
                )
            except Exception as rollback_error:
                logging_utility.error(
                    "CRITICAL: Failed to rollback Qdrant collection '%s' after API failure: %s",
                    backend_collection_name,
                    str(rollback_error),
                )
            raise api_error  # Re-raise API error

    async def _internal_add_file_to_vector_store_async(
        self,
        vector_store_id: str,
        file_path: Path,
        user_metadata: Optional[Dict[str, Any]],
    ) -> ValidationInterface.VectorStoreFileRead:  # CORRECTED Return Type Hint
        """
        Processes a file, uploads its chunks/vectors to Qdrant, and registers
        the file metadata via the API. Returns the validated file record object.
        """
        collection_name = (
            vector_store_id  # Assuming vector_store_id is used as collection name
        )

        logging_utility.info(
            "Processing file: %s for store %s", file_path, vector_store_id
        )
        try:
            # Assuming self.file_processor and its methods exist and work as expected
            processed_data = await self.file_processor.process_file(file_path)
            texts, vectors = processed_data["chunks"], processed_data["vectors"]

            if not texts or not vectors:
                logging_utility.warning(
                    "File '%s' resulted in no processable content.", file_path.name
                )
                # Depending on requirements, you might raise an error or return a specific status
                raise VectorStoreClientError(
                    f"File '{file_path.name}' resulted in no processable content."
                )

            # Prepare metadata for each chunk
            base_metadata = user_metadata or {}
            # Ensure 'source' and 'file_name' are consistently added for potential filtering/deletion
            base_metadata.update(
                {"source": str(file_path), "file_name": file_path.name}
            )
            chunk_metadata = [
                {**base_metadata, "chunk_index": i} for i in range(len(texts))
            ]

            logging_utility.info(
                "Processed file '%s' into %d chunks.", file_path.name, len(texts)
            )

        except Exception as e:
            logging_utility.error(
                "Failed to process file %s: %s", file_path, str(e), exc_info=True
            )
            # You might want to update the file status to 'failed' via API here if possible
            raise VectorStoreClientError(
                f"File processing failed for '{file_path.name}': {str(e)}"
            ) from e

        logging_utility.info(
            "Uploading %d chunks for '%s' to Qdrant collection '%s'",
            len(texts),
            file_path.name,
            collection_name,
        )
        try:
            # Assuming self.vector_manager and its methods exist and work as expected
            _ = self.vector_manager.add_to_store(
                store_name=collection_name,
                texts=texts,
                vectors=vectors,
                metadata=chunk_metadata,
                # Consider adding unique IDs for points if needed for deletion later
            )
            logging_utility.info(
                "Successfully uploaded %d chunks to Qdrant for '%s'.",
                len(texts),
                file_path.name,
            )
        except Exception as e:
            logging_utility.error(
                "Qdrant upload failed for file '%s' to collection '%s': %s",
                file_path.name,
                collection_name,
                str(e),
                exc_info=True,
            )
            # If upload fails, the file shouldn't be registered in the API.
            raise VectorStoreClientError(
                f"Vector store upload failed for '{file_path.name}': {str(e)}"
            ) from e

        # Generate a unique ID for the file record in your database
        file_record_id = f"vsf_{uuid.uuid4()}"

        # Prepare payload for the API call to register the file record
        # This should match the ValidationInterface.VectorStoreFileCreate model
        api_payload = {
            "file_id": file_record_id,
            "file_name": file_path.name,
            "file_path": str(file_path),  # Use the file path as the identifier
            "status": "completed",  # Mark as completed since Qdrant upload succeeded
            "meta_data": user_metadata or {},  # Pass along any user-provided metadata
        }

        logging_utility.info(
            "Registering file '%s' (Record ID: %s) in vector store '%s' via API",
            file_path.name,
            file_record_id,
            vector_store_id,
        )
        try:
            # Make the API call to POST /v1/vector-stores/{vector_store_id}/files
            response_data = await self._internal_request_with_retries(
                "POST", f"/v1/vector-stores/{vector_store_id}/files", json=api_payload
            )

            # Log success before validation, in case validation itself fails
            logging_utility.info(
                "Successfully registered file '%s' via API. Response keys: %s",
                file_path.name,
                (
                    list(response_data.keys())
                    if isinstance(response_data, dict)
                    else type(response_data)
                ),
            )

            # --- THIS IS THE FIX ---
            # Validate the API response against the VectorStoreFileRead model
            validated_file_record = (
                ValidationInterface.VectorStoreFileRead.model_validate(response_data)
            )
            # --- END FIX ---

            logging_utility.debug(
                "Validated API response for file registration: %s",
                validated_file_record,
            )
            return validated_file_record

        except Exception as api_error:
            # This catches errors during the API call OR during the validation of the response
            logging_utility.critical(
                "QDRANT UPLOAD SUCCEEDED for file '%s' to store '%s', BUT API registration/validation FAILED. Error: %s",
                file_path.name,
                vector_store_id,
                str(api_error),
                exc_info=True,  # Include traceback for debugging
            )

            raise api_error

    async def _internal_search_vector_store_async(
        self, vector_store_id: str, query_text: str, top_k: int, filters: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        try:
            store_info = self.retrieve_vector_store_sync(vector_store_id)
            collection_name = store_info.collection_name
        except VectorStoreClientError as e:
            logging_utility.error(
                f"Vector store {vector_store_id} not found via API: {e}"
            )
            raise

        try:
            query_vector = self.file_processor.embedding_model.encode(
                query_text
            ).tolist()
        except Exception as e:
            logging_utility.error("Failed to embed query text: %s", str(e))
            raise VectorStoreClientError(f"Query embedding failed: {str(e)}") from e

        logging_utility.info(
            "Searching Qdrant collection '%s' with top_k=%d", collection_name, top_k
        )
        try:
            search_results = self.vector_manager.query_store(
                store_name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
            )
            logging_utility.info(
                "Qdrant search completed. Found %d results.", len(search_results)
            )
            return search_results
        except Exception as e:
            logging_utility.error(
                "Qdrant search failed for collection %s: %s", collection_name, str(e)
            )
            raise VectorStoreClientError(f"Vector store search failed: {str(e)}") from e

    async def _internal_delete_vector_store_async(
        self, vector_store_id: str, permanent: bool
    ) -> Dict[str, Any]:
        collection_name = vector_store_id
        logging_utility.info(
            "Attempting to delete Qdrant collection '%s'", collection_name
        )
        qdrant_result = None
        try:
            qdrant_result = self.vector_manager.delete_store(collection_name)
            logging_utility.info(
                "Qdrant delete result for collection '%s': %s",
                collection_name,
                qdrant_result,
            )
        except Exception as e:
            logging_utility.error(
                "Qdrant collection deletion failed for '%s': %s.",
                collection_name,
                str(e),
            )
            if permanent:
                raise VectorStoreClientError(
                    f"Failed to permanently delete vector store backend: {str(e)}"
                ) from e

        logging_utility.info(
            "Calling API to %s delete vector store '%s'",
            "permanently" if permanent else "soft",
            vector_store_id,
        )
        try:
            api_response = await self._internal_request_with_retries(
                "DELETE",
                f"/v1/vector-stores/{vector_store_id}",
                params={"permanent": permanent},
            )
            logging_utility.info(
                "API delete call successful for vector store '%s'.", vector_store_id
            )
            # API returns 204, so api_response will be None. Construct success dict.
            return {
                "vector_store_id": vector_store_id,
                "status": "deleted",
                "permanent": permanent,
                "qdrant_result": qdrant_result,
            }
        except Exception as api_error:
            logging_utility.error(
                "API delete call failed for vector store '%s'. Qdrant status: %s. Error: %s",
                vector_store_id,
                qdrant_result,
                str(api_error),
            )
            raise api_error

    async def _internal_delete_file_from_vector_store_async(
        self, vector_store_id: str, file_path: str
    ) -> Dict[str, Any]:
        collection_name = vector_store_id
        logging_utility.info(
            "Attempting to delete chunks for file '%s' from Qdrant collection '%s'",
            file_path,
            collection_name,
        )
        qdrant_result = None
        try:
            qdrant_result = self.vector_manager.delete_file_from_store(
                collection_name, file_path
            )
            logging_utility.info(
                "Qdrant delete result for file '%s': %s", file_path, qdrant_result
            )
        except Exception as e:
            logging_utility.error(
                "Qdrant deletion failed for file '%s' in collection '%s': %s",
                file_path,
                collection_name,
                str(e),
            )
            raise VectorStoreClientError(
                f"Failed to delete file from vector store backend: {str(e)}"
            ) from e

        logging_utility.info(
            "Calling API to delete record for file '%s' in vector store '%s'",
            file_path,
            vector_store_id,
        )
        try:
            encoded_file_path = httpx.URL(f"/{file_path}").path[1:]
            api_response = await self._internal_request_with_retries(
                "DELETE",
                f"/v1/vector-stores/{vector_store_id}/files",
                params={"file_path": encoded_file_path},
            )
            logging_utility.info(
                "API delete call successful for file record '%s'.", file_path
            )
            # API returns 204, so api_response will be None. Construct success dict.
            return {
                "vector_store_id": vector_store_id,
                "file_path": file_path,
                "status": "deleted",
                "qdrant_result": qdrant_result,
            }
        except Exception as api_error:
            logging_utility.critical(
                "QDRANT DELETE SUCCEEDED for file '%s' in store '%s', BUT API deletion FAILED. Error: %s",
                file_path,
                vector_store_id,
                str(api_error),
            )
            raise api_error

    async def _internal_list_store_files_async(
        self, vector_store_id: str
    ) -> List[ValidationInterface.VectorStoreFileRead]:
        logging_utility.info(
            "Listing files for vector store '%s' via API", vector_store_id
        )
        try:
            response_data = await self._internal_request_with_retries(
                "GET", f"/v1/vector-stores/{vector_store_id}/files"
            )
            if not isinstance(response_data, list):
                raise VectorStoreClientError(
                    f"API returned non-list response for files: {response_data}"
                )
            return [
                ValidationInterface.VectorStoreFileRead.model_validate(item)
                for item in response_data
            ]
        except Exception as api_error:
            logging_utility.error(
                "Failed to list files for store '%s' via API: %s",
                vector_store_id,
                str(api_error),
            )
            raise api_error

    async def _internal_attach_vs_async(
        self, vector_store_id: str, assistant_id: str
    ) -> Dict[str, Any]:
        logging_utility.info(
            "Attaching vector store %s to assistant %s via API",
            vector_store_id,
            assistant_id,
        )
        # API returns {"success": True} on success (Status 200)
        return await self._internal_request_with_retries(
            "POST",
            f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/attach",
        )

    async def _internal_detach_vs_async(
        self, vector_store_id: str, assistant_id: str
    ) -> Dict[str, Any]:
        logging_utility.info(
            "Detaching vector store %s from assistant %s via API",
            vector_store_id,
            assistant_id,
        )
        # API returns {"success": True} on success (Status 200)
        return await self._internal_request_with_retries(
            "DELETE",
            f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/detach",
        )

    async def _internal_get_assistant_vs_async(
        self, assistant_id: str
    ) -> List[ValidationInterface.VectorStoreRead]:
        logging_utility.info(
            "Getting vector stores for assistant %s via API", assistant_id
        )
        response = await self._internal_request_with_retries(
            "GET", f"/v1/assistants/{assistant_id}/vector-stores"
        )
        if not isinstance(response, list):
            raise VectorStoreClientError(
                f"API returned non-list response for assistant stores: {response}"
            )
        return [
            ValidationInterface.VectorStoreRead.model_validate(item)
            for item in response
        ]

    async def _internal_get_user_vs_async(
        self, user_id: str
    ) -> List[ValidationInterface.VectorStoreRead]:
        logging_utility.info("Getting vector stores for user %s via API", user_id)
        response = await self._internal_request_with_retries(
            "GET", f"/v1/users/{user_id}/vector-stores"
        )
        if not isinstance(response, list):
            raise VectorStoreClientError(
                f"API returned non-list response for user stores: {response}"
            )
        return [
            ValidationInterface.VectorStoreRead.model_validate(item)
            for item in response
        ]

    async def _internal_retrieve_vs_async(
        self, vector_store_id: str
    ) -> ValidationInterface.VectorStoreRead:
        logging_utility.info("Retrieving vector store %s via API", vector_store_id)
        response = await self._internal_request_with_retries(
            "GET", f"/v1/vector-stores/{vector_store_id}"
        )
        return ValidationInterface.VectorStoreRead.model_validate(response)

    async def _internal_retrieve_vs_by_collection_async(
        self, collection_name: str
    ) -> ValidationInterface.VectorStoreRead:
        logging_utility.info(
            "Retrieving vector store by collection name %s via API", collection_name
        )
        response = await self._internal_request_with_retries(
            "GET",
            "/v1/vector-stores/lookup/collection",
            params={"name": collection_name},
        )
        return ValidationInterface.VectorStoreRead.model_validate(response)

    # --- NEW: Internal Async for Update File Status ---
    async def _internal_update_file_status_async(
        self,
        vector_store_id: str,
        file_id: str,
        status: ValidationInterface.StatusEnum,
        error_message: Optional[str] = None,
    ) -> ValidationInterface.VectorStoreFileRead:
        """Core async logic for updating file status via API."""
        logging_utility.info(
            f"Updating status for file '{file_id}' in store '{vector_store_id}' to '{status}' via API"
        )
        payload = VectorStoreFileUpdateStatusInput(
            status=status, error_message=error_message
        ).model_dump(exclude_none=True)
        response_data = await self._internal_request_with_retries(
            "PATCH",
            f"/v1/vector-stores/{vector_store_id}/files/{file_id}",
            json=payload,
        )
        # API returns the updated VectorStoreFileRead model
        return ValidationInterface.VectorStoreFileRead.model_validate(response_data)

    # --- Public Synchronous Methods ---

    def _run_sync(self, coro):
        """Helper to run coroutine synchronously, handling loop detection."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise VectorStoreClientError(
                    "Cannot call synchronous method from within an active asyncio event loop."
                )
        except RuntimeError:
            pass  # No loop running
        return asyncio.run(coro)

    def create_vector_store(
        self,
        name: str,
        user_id: str,
        vector_size: int = 384,
        distance_metric: str = "Cosine",
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreRead:
        """Synchronously creates a vector store."""
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
    ) -> ValidationInterface.VectorStoreFileRead:  # <--- CHANGE Return Type Hint
        """Synchronously processes and adds a file to a vector store."""
        _file_path = Path(file_path)
        if not _file_path.is_file():
            raise FileNotFoundError(f"File not found: {_file_path}")
        # The internal async method now correctly returns VectorStoreFileRead
        return self._run_sync(
            self._internal_add_file_to_vector_store_async(
                vector_store_id, _file_path, user_metadata
            )
        )

    def search_vector_store(
        self,
        vector_store_id: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronously searches a vector store."""
        return self._run_sync(
            self._internal_search_vector_store_async(
                vector_store_id, query_text, top_k, filters
            )
        )

    def delete_vector_store(
        self, vector_store_id: str, permanent: bool = False
    ) -> Dict[str, Any]:
        """Synchronously deletes a vector store."""
        return self._run_sync(
            self._internal_delete_vector_store_async(vector_store_id, permanent)
        )

    def delete_file_from_vector_store(
        self, vector_store_id: str, file_path: str
    ) -> Dict[str, Any]:
        """Synchronously deletes a file's data from a vector store."""
        return self._run_sync(
            self._internal_delete_file_from_vector_store_async(
                vector_store_id, file_path
            )
        )

    def list_store_files(
        self, vector_store_id: str
    ) -> List[ValidationInterface.VectorStoreFileRead]:
        """Synchronously lists files associated with a vector store."""
        return self._run_sync(self._internal_list_store_files_async(vector_store_id))

    # --- NEW: Public Sync for Update File Status ---
    def update_vector_store_file_status(
        self,
        vector_store_id: str,
        file_id: str,
        status: ValidationInterface.StatusEnum,
        error_message: Optional[str] = None,
    ) -> ValidationInterface.VectorStoreFileRead:
        """
        Synchronously updates the status and optionally the error message for a file record.
        """
        return self._run_sync(
            self._internal_update_file_status_async(
                vector_store_id, file_id, status, error_message
            )
        )

    def attach_vector_store_to_assistant(
        self, vector_store_id: str, assistant_id: str
    ) -> Dict[str, Any]:
        """Synchronously attaches a vector store to an assistant via API."""
        return self._run_sync(
            self._internal_attach_vs_async(vector_store_id, assistant_id)
        )

    def detach_vector_store_from_assistant(
        self, vector_store_id: str, assistant_id: str
    ) -> Dict[str, Any]:
        """Synchronously detaches a vector store from an assistant via API."""
        return self._run_sync(
            self._internal_detach_vs_async(vector_store_id, assistant_id)
        )

    def get_vector_stores_for_assistant(
        self, assistant_id: str
    ) -> List[ValidationInterface.VectorStoreRead]:
        """Synchronously gets vector stores attached to an assistant via API."""
        return self._run_sync(self._internal_get_assistant_vs_async(assistant_id))

    def get_stores_by_user(
        self, user_id: str
    ) -> List[ValidationInterface.VectorStoreRead]:
        """Synchronously gets vector stores owned by a user via API."""
        return self._run_sync(self._internal_get_user_vs_async(user_id))

    def retrieve_vector_store(
        self, vector_store_id: str
    ) -> ValidationInterface.VectorStoreRead:
        """Synchronously retrieves vector store metadata by its ID via API."""
        return self._run_sync(self._internal_retrieve_vs_async(vector_store_id))

    def retrieve_vector_store_by_collection(
        self, collection_name: str
    ) -> ValidationInterface.VectorStoreRead:
        """Synchronously retrieves vector store metadata by its collection name via API."""
        return self._run_sync(
            self._internal_retrieve_vs_by_collection_async(collection_name)
        )

    # --- Optional: Keep direct sync method if useful internally or for specific perf needs ---
    def retrieve_vector_store_sync(
        self, vector_store_id: str
    ) -> ValidationInterface.VectorStoreRead:
        """Synchronous retrieval using the sync client directly (less overhead than asyncio.run)."""
        logging_utility.info(
            "Retrieving vector store %s via sync client", vector_store_id
        )
        try:
            response = self._sync_api_client.get(f"/v1/vector-stores/{vector_store_id}")
            response.raise_for_status()
            return ValidationInterface.VectorStoreRead.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "Sync API request failed: Status %d, Response: %s",
                e.response.status_code,
                e.response.text,
            )
            raise VectorStoreClientError(
                f"API Error: {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            logging_utility.error("Failed to parse sync API response: %s", str(e))
            raise VectorStoreClientError(f"Invalid response from sync API: {e}") from e
