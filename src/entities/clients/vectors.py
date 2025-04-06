#!/usr/bin/env python
import asyncio
import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import warnings # Import warnings

import httpx
from dotenv import load_dotenv
from entities_common import UtilsInterface
from entities_common.utils import IdentifierService
from entities_common.validation import ValidationInterface

# Assuming these client classes are compatible with being instantiated here
# And their methods are either sync or async as called below.
from entities.clients.file_processor import FileProcessor
# Assuming VectorStoreManager has been updated as discussed
from entities.clients.vector_store_manager import VectorStoreManager

load_dotenv()
logging_utility = UtilsInterface.LoggingUtility()


class VectorStoreClientError(Exception):
    """Custom exception for VectorStoreClient errors."""
    pass


class VectorStoreClient:
    """
    Client for interacting with the Vector Store API and backend (e.g., Qdrant).

    Provides synchronous methods for ease of use in standard applications,
    while utilizing asynchronous operations internally for network I/O.
    """
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_store_host: Optional[str] = 'localhost',
    ):
        self.base_url = base_url or os.getenv("BASE_URL")
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.base_url:
            raise VectorStoreClientError(
                "BASE_URL must be provided either as an argument or in environment variables."
            )

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # Underlying async client for internal use
        self._async_api_client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=30.0
        )
        # Underlying sync client (kept for potential direct sync needs, like retrieve_vector_store_sync)
        self._sync_api_client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=30.0
        )

        self.vector_store_host = vector_store_host
        # Assuming VectorStoreManager methods are synchronous unless used with asyncio.to_thread
        self.vector_manager = VectorStoreManager(vector_store_host=self.vector_store_host)
        self.identifier_service = IdentifierService()
        # Assuming FileProcessor setup is synchronous, but process_file might be async
        self.file_processor = FileProcessor()

    # --- Context Managers ---
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()


    # --- Close Methods ---
    async def aclose(self):
        """Asynchronously closes the underlying HTTP clients."""
        await self._async_api_client.aclose()
        await asyncio.to_thread(self._sync_api_client.close)
        # Add closing logic for vector_manager if needed
        # await asyncio.to_thread(self.vector_manager.close)

    def close(self):
        """
        Synchronously closes the underlying HTTP clients.

        Note: Uses asyncio.run() and should not be called from within a running loop.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                warnings.warn("Calling synchronous close() from within a running event loop is problematic. Use aclose() instead.", RuntimeWarning, stacklevel=2)
                # Attempt basic sync close, may leave async client open
                try: self._sync_api_client.close()
                except Exception: pass # Ignore errors on close
                logging_utility.warning("Synchronous close called from running loop may not fully close async resources.")
                return
        except RuntimeError: # No loop running, expected case
            pass
        try:
             asyncio.run(self.aclose())
        except Exception as e:
             logging_utility.error(f"Error during client closure: {e}", exc_info=False)

    # --- Internal Async Helper Methods (Private) ---
    async def _internal_parse_response(self, response: httpx.Response) -> Any:
        try:
            response.raise_for_status()
            if response.status_code == 204: return None
            return response.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error("API request failed: Status %d, Response: %s", e.response.status_code, e.response.text)
            raise VectorStoreClientError(f"API Error: {e.response.status_code} - {e.response.text}") from e
        except Exception as e: # Includes JSONDecodeError
            logging_utility.error("Failed to parse API response: %s", str(e))
            raise VectorStoreClientError(f"Invalid response from API: {response.text}") from e

    async def _internal_request_with_retries(self, method: str, url: str, **kwargs) -> Any:
        retries = 3; last_exception = None
        for attempt in range(retries):
            try:
                response = await self._async_api_client.request(method, url, **kwargs)
                return await self._internal_parse_response(response)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_exception = e
                should_retry = isinstance(e, (httpx.TimeoutException, httpx.NetworkError)) or \
                               (isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500)
                if should_retry and attempt < retries - 1:
                    wait_time = 2**attempt
                    logging_utility.warning("Retrying request (attempt %d/%d) to %s %s after %d s. Error: %s", attempt + 1, retries, method, url, wait_time, str(e))
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logging_utility.error("API Request failed permanently after %d attempts to %s %s. Last Error: %s", attempt + 1, method, url, str(e))
                    if isinstance(e, httpx.HTTPStatusError): raise VectorStoreClientError(f"API Error: {e.response.status_code} - {e.response.text}") from e
                    else: raise VectorStoreClientError(f"API Communication Error: {str(e)}") from e
            except Exception as e: # Catch other unexpected errors like parsing
                logging_utility.error("Unexpected error during API request to %s %s: %s", method, url, str(e))
                raise VectorStoreClientError(f"Unexpected API Client Error: {str(e)}") from e
        # Should be unreachable, but satisfy linters
        raise VectorStoreClientError("Request failed after retries.") from last_exception


    # --- Internal Async Implementations ---

    async def _internal_create_vector_store_async(
        self, name: str, user_id: str, vector_size: int, distance_metric: str, config: Optional[Dict[str, Any]]
    ) -> ValidationInterface.VectorStoreRead:
        """Core async logic for creating a vector store."""
        shared_id = self.identifier_service.generate_vector_id()
        # Use the unique ID as the actual collection name for the backend
        backend_collection_name = shared_id

        logging_utility.info("Attempting to create Qdrant collection '%s'", backend_collection_name)
        try:
            # Call the manager using the correct parameter name 'collection_name'
            # Assuming vector_manager.create_store is synchronous
            qdrant_result = self.vector_manager.create_store(
                collection_name=backend_collection_name, # <<< CORRECTED Parameter Name
                vector_size=vector_size,
                distance=distance_metric.upper(),
            )
            # Optional: Check qdrant_result for success confirmation if it returns one
            # if not qdrant_result or qdrant_result.get("status") != "created":
            #     raise VectorStoreClientError(f"Failed to create Qdrant collection '{backend_collection_name}', manager returned: {qdrant_result}")

            logging_utility.info("Successfully created Qdrant collection '%s'", backend_collection_name)
        except Exception as e:
            logging_utility.error("Qdrant collection creation failed for '%s': %s", backend_collection_name, str(e))
            # Don't raise VectorStoreManager's internal errors directly, wrap them
            raise VectorStoreClientError(f"Failed to create vector store backend: {str(e)}") from e

        # If backend creation succeeded, register via API
        logging_utility.info("Registering vector store '%s' (ID: %s) via API", name, shared_id)
        db_payload = {
            "shared_id": shared_id, "name": name, "user_id": user_id,
            "vector_size": vector_size, "distance_metric": distance_metric.upper(),
            "config": config or {},
        }
        try:
            response_data = await self._internal_request_with_retries("POST", "/v1/vector-stores", json=db_payload)
            logging_utility.info("Successfully registered vector store '%s' via API", name)
            return ValidationInterface.VectorStoreRead.model_validate(response_data)
        except Exception as api_error:
            # Rollback backend creation if API registration fails
            logging_utility.error("API registration failed for store '%s' (ID: %s). Rolling back Qdrant collection. Error: %s", name, shared_id, str(api_error))
            try:
                 # Assuming vector_manager.delete_store is synchronous
                self.vector_manager.delete_store(backend_collection_name)
                logging_utility.info("Rolled back Qdrant collection '%s'", backend_collection_name)
            except Exception as rollback_error:
                logging_utility.error("CRITICAL: Failed to rollback Qdrant collection '%s' after API failure: %s", backend_collection_name, str(rollback_error))
                # Log this critical state failure prominently
            raise api_error # Re-raise the original API error

    async def _internal_add_file_to_vector_store_async(
        self, vector_store_id: str, file_path: Path, user_metadata: Optional[Dict[str, Any]] # Removed unused params
    ) -> ValidationInterface.VectorStoreRead:
        """Core async logic for adding a file."""
        # Collection name in Qdrant is the unique vector_store_id
        collection_name = vector_store_id

        logging_utility.info("Processing file: %s", file_path)
        try:
            # Assumes file_processor.process_file is async
            processed_data = await self.file_processor.process_file(file_path)
            texts, vectors = processed_data["chunks"], processed_data["vectors"]
            if not texts or not vectors:
                logging_utility.warning(f"File processing yielded no chunks/vectors for {file_path}. Skipping upload & registration.")
                # What should be returned here? Maybe the existing store state?
                # Or raise an error? Let's raise for now.
                raise VectorStoreClientError(f"File '{file_path.name}' resulted in no processable content.")

            base_metadata = user_metadata or {}
            # Ensure 'source' and 'file_name' are added for Qdrant filtering/identification
            base_metadata.update({"source": str(file_path), "file_name": file_path.name})
            chunk_metadata = [{**base_metadata, "chunk_index": i} for i in range(len(texts))]
            logging_utility.info("Processed file '%s' into %d chunks.", file_path.name, len(texts))
        except Exception as e:
            logging_utility.error("Failed to process file %s: %s", file_path, str(e))
            raise VectorStoreClientError(f"File processing failed: {str(e)}") from e

        logging_utility.info("Uploading %d chunks for '%s' to Qdrant collection '%s'", len(texts), file_path.name, collection_name)
        try:
            # Assuming vector_manager.add_to_store is synchronous
            qdrant_result = self.vector_manager.add_to_store(
                collection_name=collection_name, # Use collection_name for Qdrant
                texts=texts, vectors=vectors, metadata=chunk_metadata
            )
            logging_utility.info("Successfully uploaded chunks to Qdrant for '%s'. Result: %s", file_path.name, qdrant_result)
        except Exception as e:
            logging_utility.error("Qdrant upload failed for file %s to collection %s: %s", file_path.name, collection_name, str(e))
            raise VectorStoreClientError(f"Vector store upload failed: {str(e)}") from e

        # If Qdrant upload succeeded, register file via API
        file_record_id = f"vsf_{uuid.uuid4()}" # Generate unique ID for the DB record
        api_payload = {
            "file_id": file_record_id, # ID for the VectorStoreFile record
            "file_name": file_path.name,
            "file_path": str(file_path), # Store the path used in Qdrant metadata['source']
            "status": "completed", # Assuming success if we reached here
            "meta_data": user_metadata or {},
        }
        logging_utility.info("Registering file '%s' (Record ID: %s) in vector store '%s' via API", file_path.name, file_record_id, vector_store_id)
        try:
            # Correct API endpoint: POST /v1/vector-stores/{vector_store_id}/files
            response_data = await self._internal_request_with_retries("POST", f"/v1/vector-stores/{vector_store_id}/files", json=api_payload)
            logging_utility.info("Successfully registered file '%s' via API.", file_path.name)
            # Assuming API returns the updated VectorStore state
            return ValidationInterface.VectorStoreRead.model_validate(response_data)
        except Exception as api_error:
            logging_utility.critical("QDRANT UPLOAD SUCCEEDED for file '%s' to store '%s', BUT API registration FAILED. Manual reconciliation may be required. Error: %s", file_path.name, vector_store_id, str(api_error))
            # Consider attempting deletion from Qdrant here, though it adds complexity
            raise api_error # Re-raise API error

    async def _internal_search_vector_store_async(
        self, vector_store_id: str, query_text: str, top_k: int, filters: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Core async logic for searching."""
        try:
            # Use the direct sync method here for efficiency before async embedding
            store_info = self.retrieve_vector_store_sync(vector_store_id)
            collection_name = store_info.collection_name
        except VectorStoreClientError as e:
            # Catch specific client error if store not found via sync call
            logging_utility.error(f"Vector store {vector_store_id} not found via API: {e}")
            raise # Re-raise the client error

        try:
            # Assuming embedding_model.encode is sync & CPU-bound: run in thread if slow
            # query_vector = await asyncio.to_thread(self.file_processor.embedding_model.encode(query_text).tolist)
            query_vector = self.file_processor.embedding_model.encode(query_text).tolist() # Direct call
        except Exception as e:
            logging_utility.error("Failed to embed query text: %s", str(e))
            raise VectorStoreClientError(f"Query embedding failed: {str(e)}") from e

        logging_utility.info("Searching Qdrant collection '%s' with top_k=%d", collection_name, top_k)
        try:
            # Assuming vector_manager.query_store is sync
            # If blocking IO: search_results = await asyncio.to_thread(self.vector_manager.query_store, ...)
            search_results = self.vector_manager.query_store(
                 collection_name=collection_name, # Use correct param name for manager
                 query_vector=query_vector, top_k=top_k, filters=filters
             )
            logging_utility.info("Qdrant search completed. Found %d results.", len(search_results))
            return search_results # Return raw results from Qdrant
        except Exception as e:
            logging_utility.error("Qdrant search failed for collection %s: %s", collection_name, str(e))
            raise VectorStoreClientError(f"Vector store search failed: {str(e)}") from e

    async def _internal_delete_vector_store_async(
        self, vector_store_id: str, permanent: bool
    ) -> Dict[str, Any]:
        """Core async logic for deleting a store."""
        # Collection name in Qdrant is the unique vector_store_id
        collection_name = vector_store_id

        logging_utility.info("Attempting to delete Qdrant collection '%s'", collection_name)
        qdrant_result = None
        try:
             # Assuming vector_manager.delete_store is sync
            qdrant_result = self.vector_manager.delete_store(collection_name)
            logging_utility.info("Qdrant delete result for collection '%s': %s", collection_name, qdrant_result)
        except Exception as e:
            # Log error but proceed to API call if not permanent delete, otherwise raise
            logging_utility.error("Qdrant collection deletion failed for '%s': %s.", collection_name, str(e))
            if permanent: raise VectorStoreClientError(f"Failed to permanently delete vector store backend: {str(e)}") from e
            # If soft delete, we still want to mark it deleted in the DB

        logging_utility.info("Calling API to %s delete vector store '%s'", "permanently" if permanent else "soft", vector_store_id)
        try:
            api_response = await self._internal_request_with_retries(
                "DELETE", f"/v1/vector-stores/{vector_store_id}", params={"permanent": permanent}
            )
            logging_utility.info("API delete call successful for vector store '%s'.", vector_store_id)
            return {"vector_store_id": vector_store_id, "status": "deleted", "permanent": permanent, "qdrant_result": qdrant_result, "api_result": api_response}
        except Exception as api_error:
            logging_utility.error("API delete call failed for vector store '%s'. Qdrant status: %s. Error: %s", vector_store_id, qdrant_result, str(api_error))
            # If Qdrant delete succeeded (or was skipped for soft delete) but API failed, DB state is inconsistent.
            raise api_error # Re-raise API error

    async def _internal_delete_file_from_vector_store_async(
        self, vector_store_id: str, file_path: str
    ) -> Dict[str, Any]:
        """Core async logic for deleting a file."""
        # Collection name in Qdrant is the unique vector_store_id
        collection_name = vector_store_id

        logging_utility.info("Attempting to delete chunks for file '%s' from Qdrant collection '%s'", file_path, collection_name)
        qdrant_result = None
        try:
             # Assuming vector_manager.delete_file_from_store is sync
            qdrant_result = self.vector_manager.delete_file_from_store(collection_name, file_path)
            logging_utility.info("Qdrant delete result for file '%s': %s", file_path, qdrant_result)
        except Exception as e:
            # If Qdrant deletion fails, we should not proceed to delete the DB record
            logging_utility.error("Qdrant deletion failed for file '%s' in collection '%s': %s", file_path, collection_name, str(e))
            raise VectorStoreClientError(f"Failed to delete file from vector store backend: {str(e)}") from e

        # If Qdrant delete succeeded, call API to delete the DB record
        logging_utility.info("Calling API to delete record for file '%s' in vector store '%s'", file_path, vector_store_id)
        try:
            # URL encode file_path just in case it has special characters
            encoded_file_path = httpx.URL(f"/{file_path}").path[1:] # Basic encoding trick
            api_response = await self._internal_request_with_retries(
                "DELETE", f"/v1/vector-stores/{vector_store_id}/files", params={"file_path": encoded_file_path}
            )
            logging_utility.info("API delete call successful for file record '%s'.", file_path)
            return {"vector_store_id": vector_store_id, "file_path": file_path, "status": "deleted", "qdrant_result": qdrant_result, "api_result": api_response}
        except Exception as api_error:
            logging_utility.critical("QDRANT DELETE SUCCEEDED for file '%s' in store '%s', BUT API deletion FAILED. DB file record may be orphaned. Error: %s", file_path, vector_store_id, str(api_error))
            raise api_error # Re-raise API error

    async def _internal_list_store_files_async(
        self, vector_store_id: str
    ) -> List[ValidationInterface.VectorStoreFileRead]:
        """Core async logic for listing files via API."""
        logging_utility.info("Listing files for vector store '%s' via API", vector_store_id)
        try:
            response_data = await self._internal_request_with_retries("GET", f"/v1/vector-stores/{vector_store_id}/files")
            # Ensure the response is a list before validating
            if not isinstance(response_data, list):
                 raise VectorStoreClientError(f"API returned non-list response for files: {response_data}")
            return [ValidationInterface.VectorStoreFileRead.model_validate(item) for item in response_data]
        except Exception as api_error:
            logging_utility.error("Failed to list files for store '%s' via API: %s", vector_store_id, str(api_error))
            raise api_error # Re-raise the original API error

    async def _internal_attach_vs_async(self, vector_store_id: str, assistant_id: str) -> Dict[str, Any]:
        logging_utility.info("Attaching vector store %s to assistant %s via API", vector_store_id, assistant_id)
        return await self._internal_request_with_retries("POST", f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/attach")

    async def _internal_detach_vs_async(self, vector_store_id: str, assistant_id: str) -> Dict[str, Any]:
        logging_utility.info("Detaching vector store %s from assistant %s via API", vector_store_id, assistant_id)
        return await self._internal_request_with_retries("DELETE", f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/detach")

    async def _internal_get_assistant_vs_async(self, assistant_id: str) -> List[ValidationInterface.VectorStoreRead]:
        logging_utility.info("Getting vector stores for assistant %s via API", assistant_id)
        response = await self._internal_request_with_retries("GET", f"/v1/assistants/{assistant_id}/vector-stores")
        if not isinstance(response, list): raise VectorStoreClientError(f"API returned non-list response for assistant stores: {response}")
        return [ValidationInterface.VectorStoreRead.model_validate(item) for item in response]

    async def _internal_get_user_vs_async(self, user_id: str) -> List[ValidationInterface.VectorStoreRead]:
        logging_utility.info("Getting vector stores for user %s via API", user_id)
        response = await self._internal_request_with_retries("GET", f"/v1/users/{user_id}/vector-stores")
        if not isinstance(response, list): raise VectorStoreClientError(f"API returned non-list response for user stores: {response}")
        return [ValidationInterface.VectorStoreRead.model_validate(item) for item in response]

    async def _internal_retrieve_vs_async(self, vector_store_id: str) -> ValidationInterface.VectorStoreRead:
        logging_utility.info("Retrieving vector store %s via API", vector_store_id)
        response = await self._internal_request_with_retries("GET", f"/v1/vector-stores/{vector_store_id}")
        return ValidationInterface.VectorStoreRead.model_validate(response)

    async def _internal_retrieve_vs_by_collection_async(self, collection_name: str) -> ValidationInterface.VectorStoreRead:
        logging_utility.info("Retrieving vector store by collection name %s via API", collection_name)
        response = await self._internal_request_with_retries("GET", "/v1/vector-stores/lookup/collection", params={"name": collection_name})
        return ValidationInterface.VectorStoreRead.model_validate(response)


    # --- Public Synchronous Methods ---

    def _run_sync(self, coro):
        """Helper to run coroutine synchronously, handling loop detection."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # This is the problematic case for a library designed for sync use.
                # Raising an error is the safest way to prevent unexpected blocking or errors.
                raise VectorStoreClientError(
                    "Cannot call synchronous method from within an active asyncio event loop. "
                    "Consider using the client in a separate thread or process, or refactor the calling code."
                    # If an async version of the method exists (e.g., client.create_vector_store_async), suggest that.
                )
        except RuntimeError:  # No loop running, safe to use asyncio.run
            pass
        # If no loop was running, create one, run the coroutine, and close the loop.
        return asyncio.run(coro)


    def create_vector_store(
        self,
        name: str,
        user_id: str,
        vector_size: int = 384,
        distance_metric: str = "Cosine",
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreRead:
        """
        Synchronously creates a vector store.
        Creates collection in backend (e.g., Qdrant) and registers metadata via API.
        """
        return self._run_sync(self._internal_create_vector_store_async(name, user_id, vector_size, distance_metric, config))

    def add_file_to_vector_store(
        self,
        vector_store_id: str,
        file_path: Union[str, Path],
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreRead:
        """
        Synchronously processes a file, uploads chunks to the vector store backend,
        and registers the file association via API.
        """
        _file_path = Path(file_path)
        if not _file_path.is_file(): raise FileNotFoundError(f"File not found: {_file_path}")
        # Removed unused parameters (chunk_size, embedding_model_name) from the call signature
        return self._run_sync(self._internal_add_file_to_vector_store_async(vector_store_id, _file_path, user_metadata))

    def search_vector_store(
        self,
        vector_store_id: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronously performs semantic search against the vector store backend.
        """
        return self._run_sync(self._internal_search_vector_store_async(vector_store_id, query_text, top_k, filters))

    def delete_vector_store(self, vector_store_id: str, permanent: bool = False) -> Dict[str, Any]:
        """
        Synchronously deletes a vector store.
        Deletes collection from backend (e.g., Qdrant) and deletes/marks metadata via API.
        """
        return self._run_sync(self._internal_delete_vector_store_async(vector_store_id, permanent))

    def delete_file_from_vector_store(self, vector_store_id: str, file_path: str) -> Dict[str, Any]:
        """
        Synchronously deletes a file's chunks from the backend and its metadata via API.
        Uses file_path to identify data in both systems.
        """
        return self._run_sync(self._internal_delete_file_from_vector_store_async(vector_store_id, file_path))

    def list_store_files(self, vector_store_id: str) -> List[ValidationInterface.VectorStoreFileRead]:
        """
        Synchronously lists files associated with a vector store by querying the API.
        """
        return self._run_sync(self._internal_list_store_files_async(vector_store_id))

    def attach_vector_store_to_assistant(self, vector_store_id: str, assistant_id: str) -> Dict[str, Any]:
        """Synchronously attaches a vector store to an assistant via API."""
        return self._run_sync(self._internal_attach_vs_async(vector_store_id, assistant_id))

    def detach_vector_store_from_assistant(self, vector_store_id: str, assistant_id: str) -> Dict[str, Any]:
        """Synchronously detaches a vector store from an assistant via API."""
        return self._run_sync(self._internal_detach_vs_async(vector_store_id, assistant_id))

    def get_vector_stores_for_assistant(self, assistant_id: str) -> List[ValidationInterface.VectorStoreRead]:
        """Synchronously gets vector stores attached to an assistant via API."""
        return self._run_sync(self._internal_get_assistant_vs_async(assistant_id))

    def get_stores_by_user(self, user_id: str) -> List[ValidationInterface.VectorStoreRead]:
        """Synchronously gets vector stores owned by a user via API."""
        return self._run_sync(self._internal_get_user_vs_async(user_id))

    def retrieve_vector_store(self, vector_store_id: str) -> ValidationInterface.VectorStoreRead:
        """Synchronously retrieves vector store metadata by its ID via API."""
        # Can optimize slightly by calling the existing sync method if preferred
        # return self.retrieve_vector_store_sync(vector_store_id)
        # Or stick to the pattern:
        return self._run_sync(self._internal_retrieve_vs_async(vector_store_id))

    def retrieve_vector_store_by_collection(self, collection_name: str) -> ValidationInterface.VectorStoreRead:
        """Synchronously retrieves vector store metadata by its collection name via API."""
        return self._run_sync(self._internal_retrieve_vs_by_collection_async(collection_name))

    # --- Optional: Keep direct sync method if useful internally or for specific perf needs ---
    def retrieve_vector_store_sync(self, vector_store_id: str) -> ValidationInterface.VectorStoreRead:
        """Synchronous retrieval using the sync client directly (less overhead than asyncio.run)."""
        logging_utility.info("Retrieving vector store %s via sync client", vector_store_id)
        try:
            # Use internal client ref
            response = self._sync_api_client.get(f"/v1/vector-stores/{vector_store_id}")
            response.raise_for_status()
            return ValidationInterface.VectorStoreRead.model_validate(response.json())
        except httpx.HTTPStatusError as e:
             logging_utility.error("Sync API request failed: Status %d, Response: %s", e.response.status_code, e.response.text)
             raise VectorStoreClientError(f"API Error: {e.response.status_code} - {e.response.text}") from e
        except Exception as e: # Includes JSONDecodeError
             logging_utility.error("Failed to parse sync API response: %s", str(e))
             raise VectorStoreClientError(f"Invalid response from sync API: {e}") from e
