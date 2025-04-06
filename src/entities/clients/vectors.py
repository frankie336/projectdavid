#!/usr/bin/env python
import asyncio
import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

import httpx
from dotenv import load_dotenv
from entities_common import UtilsInterface
from entities_common.utils import IdentifierService
from entities_common.validation import ValidationInterface

# Assuming these client classes are compatible with being instantiated here
# And their methods are either sync or async as called below.
from entities.clients.file_processor import FileProcessor
from entities.clients.vector_store_manager import VectorStoreManager

load_dotenv()
logging_utility = UtilsInterface.LoggingUtility()


class VectorStoreClientError(Exception):
    """Custom exception for VectorStoreClient errors."""

    pass


class VectorStoreClient:
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
        # Main client for async operations
        self.api_client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=30.0
        )
        # Sync client retained for specific sync methods if ever needed directly
        self.sync_api_client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=30.0
        )

        self.vector_store_host = vector_store_host
        self.vector_manager = VectorStoreManager(vector_store_host=self.vector_store_host)
        self.identifier_service = IdentifierService()
        # FileProcessor needs careful consideration: if its methods are async,
        # they work fine within the _async methods. If sync, also fine.
        self.file_processor = FileProcessor()

    # --- Async Close Method ---
    async def close_async(self):
        """Asynchronously closes the HTTP clients."""
        await self.api_client.aclose()
        # Run sync close in executor to avoid blocking if called from async context
        # Though usually close is called when the loop is stopping anyway.
        await asyncio.to_thread(self.sync_api_client.close)

    # --- Sync Close Method ---
    def close(self):
        """Synchronously closes the HTTP clients."""
        # Need to run the async close method
        try:
            # Check if an event loop is already running. If so, schedule and wait.
            loop = asyncio.get_running_loop()
            # This scenario is complex; ideally close() is called outside a running loop.
            # If called inside, it might block. Using asyncio.run() is simpler
            # if we assume close() is called from a purely sync context.
            asyncio.run(self.close_async())
        except RuntimeError: # No running event loop
             asyncio.run(self.close_async())


    # --- Internal Helper Methods (Remain Async) ---
    async def _parse_response(self, response: httpx.Response) -> Any:
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
            raise VectorStoreClientError(f"Invalid response from API: {response.text}") from e

    async def _request_with_retries(
        self, method: str, url: str, **kwargs
    ) -> Any:
        """Internal async request helper using the async client."""
        retries = 3
        last_exception = None

        for attempt in range(retries):
            try:
                # Always use the async client here
                response = await self.api_client.request(method, url, **kwargs)
                return await self._parse_response(response) # Parse response remains async
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_exception = e
                should_retry = isinstance(e, (httpx.TimeoutException, httpx.NetworkError)) or (
                    isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500
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
                    await asyncio.sleep(wait_time) # Async sleep
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
                        raise VectorStoreClientError(f"API Communication Error: {str(e)}") from e
            except Exception as e:
                logging_utility.error(
                    "Unexpected error during API request to %s %s: %s", method, url, str(e)
                )
                raise VectorStoreClientError(f"Unexpected API Client Error: {str(e)}") from e
        raise VectorStoreClientError("Request failed after retries.") from last_exception


    # --- Vector Store CRUD ---

    # Async Implementation
    async def _create_vector_store_async(
        self,
        name: str,
        user_id: str,
        vector_size: int = 384,
        distance_metric: str = "Cosine",
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreRead:
        shared_id = self.identifier_service.generate_vector_id()
        collection_name = shared_id

        logging_utility.info("Attempting to create Qdrant collection '%s'", collection_name)
        try:
            # Assuming vector_manager methods are synchronous
            qdrant_success = self.vector_manager.create_store(
                store_name=collection_name,
                vector_size=vector_size,
                distance=distance_metric.upper(),
            )
            if not qdrant_success:
                raise VectorStoreClientError(f"Failed to create Qdrant collection '{collection_name}'")
            logging_utility.info("Successfully created Qdrant collection '%s'", collection_name)
        except Exception as e:
            logging_utility.error("Qdrant collection creation failed for '%s': %s", collection_name, str(e))
            raise VectorStoreClientError(f"Failed to create vector store backend: {str(e)}") from e

        logging_utility.info("Registering vector store '%s' (ID: %s) via API", name, shared_id)
        db_payload = {
            "shared_id": shared_id, "name": name, "user_id": user_id,
            "vector_size": vector_size, "distance_metric": distance_metric.upper(),
            "config": config or {},
        }
        try:
            response_data = await self._request_with_retries("POST", "/v1/vector-stores", json=db_payload)
            logging_utility.info("Successfully registered vector store '%s' via API", name)
            return ValidationInterface.VectorStoreRead.model_validate(response_data)
        except Exception as api_error:
            logging_utility.error("API registration failed for store '%s' (ID: %s). Rolling back. Error: %s", name, shared_id, str(api_error))
            try:
                # vector_manager.delete_store is sync
                self.vector_manager.delete_store(collection_name)
                logging_utility.info("Rolled back Qdrant collection '%s'", collection_name)
            except Exception as rollback_error:
                logging_utility.error("Failed to rollback Qdrant collection '%s': %s", collection_name, str(rollback_error))
            raise api_error

    # Sync Wrapper
    def create_vector_store(
        self,
        name: str,
        user_id: str,
        vector_size: int = 384,
        distance_metric: str = "Cosine",
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreRead:
        """Synchronously creates a vector store."""
        return asyncio.run(self._create_vector_store_async(name, user_id, vector_size, distance_metric, config))

    # --- Add File ---

    # Async Implementation
    async def _add_file_to_vector_store_async(
        self,
        vector_store_id: str,
        file_path: Union[str, Path],
        chunk_size: int = 512,
        embedding_model_name: str = "paraphrase-MiniLM-L6-v2",
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreRead:
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        collection_name = vector_store_id # Assuming ID is collection name

        logging_utility.info("Processing file: %s", file_path)
        try:
            # Assumes file_processor.process_file is async
            processed_data = await self.file_processor.process_file(file_path)
            texts = processed_data["chunks"]
            vectors = processed_data["vectors"]
            base_metadata = user_metadata or {}
            base_metadata["source"] = str(file_path)
            base_metadata["file_name"] = file_path.name
            chunk_metadata = [{**base_metadata, "chunk_index": i} for i in range(len(texts))]
            logging_utility.info("Processed file '%s' into %d chunks.", file_path.name, len(texts))
        except Exception as e:
            logging_utility.error("Failed to process file %s: %s", file_path, str(e))
            raise VectorStoreClientError(f"File processing failed: {str(e)}") from e

        logging_utility.info("Uploading %d chunks for '%s' to Qdrant collection '%s'", len(texts), file_path.name, collection_name)
        try:
            # Assuming vector_manager.add_to_store is synchronous
            qdrant_result = self.vector_manager.add_to_store(
                store_name=collection_name, texts=texts, vectors=vectors, metadata=chunk_metadata
            )
            logging_utility.info("Successfully uploaded chunks to Qdrant for '%s'. Result: %s", file_path.name, qdrant_result)
        except Exception as e:
            logging_utility.error("Qdrant upload failed for file %s to collection %s: %s", file_path.name, collection_name, str(e))
            raise VectorStoreClientError(f"Vector store upload failed: {str(e)}") from e

        file_record_id = f"vsf_{uuid.uuid4()}"
        api_payload = {
            "file_id": file_record_id, "file_name": file_path.name, "file_path": str(file_path),
            "status": "completed", "meta_data": user_metadata or {},
        }
        logging_utility.info("Registering file '%s' (ID: %s) in vector store '%s' via API", file_path.name, file_record_id, vector_store_id)
        try:
            response_data = await self._request_with_retries("POST", f"/v1/vector-stores/{vector_store_id}/files", json=api_payload)
            logging_utility.info("Successfully registered file '%s' via API.", file_path.name)
            # Assuming the response here is the VectorStore *after* file addition update
            # Adjust validation if API returns file info or confirmation instead
            return ValidationInterface.VectorStoreRead.model_validate(response_data)
        except Exception as api_error:
            logging_utility.critical("QDRANT UPLOAD SUCCEEDED for file '%s' to store '%s', BUT API registration FAILED. Error: %s", file_path.name, vector_store_id, str(api_error))
            raise api_error

    # Sync Wrapper
    def add_file_to_vector_store(
        self,
        vector_store_id: str,
        file_path: Union[str, Path],
        chunk_size: int = 512,
        embedding_model_name: str = "paraphrase-MiniLM-L6-v2",
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreRead:
        """Synchronously adds a file to a vector store."""
        return asyncio.run(self._add_file_to_vector_store_async(vector_store_id, file_path, chunk_size, embedding_model_name, user_metadata))


    # --- Search ---

    # Async Implementation
    async def _search_vector_store_async(
        self,
        vector_store_id: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        # Get store info sync first to avoid await before embedding if possible
        try:
            store_info = self.retrieve_vector_store_sync(vector_store_id)
            collection_name = store_info.collection_name
        except VectorStoreClientError:
            logging_utility.error(f"Vector store {vector_store_id} not found via API.")
            raise

        try:
            # Assuming embedding_model.encode is synchronous CPU-bound work
            # If it were async IO, use await asyncio.to_thread(...)
             query_vector = self.file_processor.embedding_model.encode(query_text).tolist()
            # query_vector = await asyncio.to_thread(self.file_processor.embedding_model.encode(query_text).tolist) # If encode is blocking
        except Exception as e:
            logging_utility.error("Failed to embed query text: %s", str(e))
            raise VectorStoreClientError(f"Query embedding failed: {str(e)}") from e

        logging_utility.info("Searching Qdrant collection '%s' with top_k=%d", collection_name, top_k)
        try:
            # Assuming vector_manager.query_store is synchronous
            search_results = self.vector_manager.query_store(
                store_name=collection_name, query_vector=query_vector, top_k=top_k, filters=filters
            )
            # search_results = await asyncio.to_thread(self.vector_manager.query_store, ...) # If query_store is blocking IO
            logging_utility.info("Qdrant search completed. Found %d results.", len(search_results))
            return search_results
        except Exception as e:
            logging_utility.error("Qdrant search failed for collection %s: %s", collection_name, str(e))
            raise VectorStoreClientError(f"Vector store search failed: {str(e)}") from e

    # Sync Wrapper
    def search_vector_store(
        self,
        vector_store_id: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronously searches a vector store."""
        return asyncio.run(self._search_vector_store_async(vector_store_id, query_text, top_k, filters))

    # --- Delete Store ---

    # Async Implementation
    async def _delete_vector_store_async(
        self, vector_store_id: str, permanent: bool = False
    ) -> Dict[str, Any]:
        collection_name = vector_store_id

        logging_utility.info("Attempting to delete Qdrant collection '%s'", collection_name)
        qdrant_result = None
        try:
            # Assuming vector_manager.delete_store is sync
            qdrant_result = self.vector_manager.delete_store(collection_name)
            logging_utility.info("Qdrant delete result for collection '%s': %s", collection_name, qdrant_result)
        except Exception as e:
            logging_utility.error("Qdrant collection deletion failed for '%s': %s.", collection_name, str(e))
            if permanent:
                raise VectorStoreClientError(f"Failed to delete vector store backend: {str(e)}") from e

        logging_utility.info("Calling API to %s delete vector store '%s'", "permanently" if permanent else "soft", vector_store_id)
        try:
            api_response = await self._request_with_retries(
                "DELETE", f"/v1/vector-stores/{vector_store_id}", params={"permanent": permanent}
            )
            logging_utility.info("API delete call successful for vector store '%s'.", vector_store_id)
            return {"vector_store_id": vector_store_id, "status": "deleted", "permanent": permanent, "qdrant_result": qdrant_result, "api_result": api_response}
        except Exception as api_error:
            logging_utility.error("API delete call failed for vector store '%s'. Qdrant status: %s. Error: %s", vector_store_id, qdrant_result, str(api_error))
            raise api_error

    # Sync Wrapper
    def delete_vector_store(self, vector_store_id: str, permanent: bool = False) -> Dict[str, Any]:
        """Synchronously deletes a vector store."""
        return asyncio.run(self._delete_vector_store_async(vector_store_id, permanent))


    # --- Delete File ---

    # Async Implementation
    async def _delete_file_from_vector_store_async(
        self, vector_store_id: str, file_path: str
    ) -> Dict[str, Any]:
        collection_name = vector_store_id

        logging_utility.info("Attempting to delete chunks for file '%s' from Qdrant collection '%s'", file_path, collection_name)
        qdrant_result = None
        try:
            # Assuming vector_manager.delete_file_from_store is sync
            qdrant_result = self.vector_manager.delete_file_from_store(collection_name, file_path)
            logging_utility.info("Qdrant delete result for file '%s': %s", file_path, qdrant_result)
        except Exception as e:
            logging_utility.error("Qdrant deletion failed for file '%s' in collection '%s': %s", file_path, collection_name, str(e))
            raise VectorStoreClientError(f"Failed to delete file from vector store backend: {str(e)}") from e

        logging_utility.info("Calling API to delete record for file '%s' in vector store '%s'", file_path, vector_store_id)
        try:
            encoded_file_path = httpx.URL(file_path).path
            api_response = await self._request_with_retries(
                "DELETE", f"/v1/vector-stores/{vector_store_id}/files", params={"file_path": encoded_file_path}
            )
            logging_utility.info("API delete call successful for file record '%s'.", file_path)
            return {"vector_store_id": vector_store_id, "file_path": file_path, "status": "deleted", "qdrant_result": qdrant_result, "api_result": api_response}
        except Exception as api_error:
            logging_utility.critical("QDRANT DELETE SUCCEEDED for file '%s' in store '%s', BUT API deletion FAILED. Error: %s", file_path, vector_store_id, str(api_error))
            raise api_error

    # Sync Wrapper
    def delete_file_from_vector_store(self, vector_store_id: str, file_path: str) -> Dict[str, Any]:
        """Synchronously deletes a file's data from a vector store."""
        return asyncio.run(self._delete_file_from_vector_store_async(vector_store_id, file_path))


    # --- List Files ---

    # Async Implementation
    async def _list_store_files_async(
        self, vector_store_id: str
    ) -> List[ValidationInterface.VectorStoreFileRead]: # Adjusted return type hint
        """Lists files associated with a vector store by querying the API (DB)."""
        logging_utility.info("Listing files for vector store '%s' via API", vector_store_id)
        try:
            response_data = await self._request_with_retries("GET", f"/v1/vector-stores/{vector_store_id}/files")
            # Adjust validation based on the actual API response for listing files
            return [ValidationInterface.VectorStoreFileRead.model_validate(item) for item in response_data]
        except Exception as api_error:
            logging_utility.error("Failed to list files for store '%s' via API: %s", vector_store_id, str(api_error))
            raise api_error # Re-raise the original API error

    # Sync Wrapper
    def list_store_files(self, vector_store_id: str) -> List[ValidationInterface.VectorStoreFileRead]:
        """Synchronously lists files associated with a vector store."""
        return asyncio.run(self._list_store_files_async(vector_store_id))


    # --- Assistant & User Methods (Repeat the pattern) ---

    # Attach Store Async
    async def _attach_vector_store_to_assistant_async(
        self, vector_store_id: str, assistant_id: str
    ) -> Dict[str, Any]: # Return type might be simple confirmation { "success": True }
        logging_utility.info("Attaching vector store %s to assistant %s via API", vector_store_id, assistant_id)
        response = await self._request_with_retries("POST", f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/attach")
        return response # Return the parsed JSON response

    # Attach Store Sync
    def attach_vector_store_to_assistant(
        self, vector_store_id: str, assistant_id: str
    ) -> Dict[str, Any]:
        """Synchronously attaches a vector store to an assistant."""
        return asyncio.run(self._attach_vector_store_to_assistant_async(vector_store_id, assistant_id))

    # Detach Store Async
    async def _detach_vector_store_from_assistant_async(
        self, vector_store_id: str, assistant_id: str
    ) -> Dict[str, Any]: # Return type might be simple confirmation { "success": True }
        logging_utility.info("Detaching vector store %s from assistant %s via API", vector_store_id, assistant_id)
        response = await self._request_with_retries("DELETE", f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/detach")
        return response # Return the parsed JSON response

    # Detach Store Sync
    def detach_vector_store_from_assistant(
        self, vector_store_id: str, assistant_id: str
    ) -> Dict[str, Any]:
        """Synchronously detaches a vector store from an assistant."""
        return asyncio.run(self._detach_vector_store_from_assistant_async(vector_store_id, assistant_id))

    # Get Stores for Assistant Async
    async def _get_vector_stores_for_assistant_async(
        self, assistant_id: str
    ) -> List[ValidationInterface.VectorStoreRead]:
        logging_utility.info("Getting vector stores for assistant %s via API", assistant_id)
        response = await self._request_with_retries("GET", f"/v1/assistants/{assistant_id}/vector-stores")
        return [ValidationInterface.VectorStoreRead.model_validate(item) for item in response]

    # Get Stores for Assistant Sync
    def get_vector_stores_for_assistant(
        self, assistant_id: str
    ) -> List[ValidationInterface.VectorStoreRead]:
        """Synchronously gets vector stores attached to an assistant."""
        return asyncio.run(self._get_vector_stores_for_assistant_async(assistant_id))

    # Get Stores by User Async
    async def _get_stores_by_user_async(self, user_id: str) -> List[ValidationInterface.VectorStoreRead]:
        logging_utility.info("Getting vector stores for user %s via API", user_id)
        response = await self._request_with_retries("GET", f"/v1/users/{user_id}/vector-stores")
        return [ValidationInterface.VectorStoreRead.model_validate(item) for item in response]

    # Get Stores by User Sync
    def get_stores_by_user(self, user_id: str) -> List[ValidationInterface.VectorStoreRead]:
        """Synchronously gets vector stores owned by a user."""
        return asyncio.run(self._get_stores_by_user_async(user_id))

    # Retrieve Store Async
    async def _retrieve_vector_store_async(
        self, vector_store_id: str
    ) -> ValidationInterface.VectorStoreRead:
        logging_utility.info("Retrieving vector store %s via API", vector_store_id)
        response = await self._request_with_retries("GET", f"/v1/vector-stores/{vector_store_id}")
        return ValidationInterface.VectorStoreRead.model_validate(response)

    # Retrieve Store Sync
    def retrieve_vector_store(
        self, vector_store_id: str
    ) -> ValidationInterface.VectorStoreRead:
        """Synchronously retrieves vector store metadata by its ID."""
        # Can optimize slightly by calling the existing sync method if preferred
        # return self.retrieve_vector_store_sync(vector_store_id)
        # Or stick to the pattern:
        return asyncio.run(self._retrieve_vector_store_async(vector_store_id))

    # Keep the original specific sync method if useful for internal calls
    def retrieve_vector_store_sync(
        self, vector_store_id: str
    ) -> ValidationInterface.VectorStoreRead:
        """Synchronous version using the sync client directly."""
        logging_utility.info("Retrieving vector store %s via sync API", vector_store_id)
        try:
            response = self.sync_api_client.get(f"{self.base_url}/v1/vector-stores/{vector_store_id}")
            response.raise_for_status()
            return ValidationInterface.VectorStoreRead.model_validate(response.json())
        except httpx.HTTPStatusError as e:
             logging_utility.error("Sync API request failed: Status %d, Response: %s", e.response.status_code, e.response.text)
             raise VectorStoreClientError(f"API Error: {e.response.status_code} - {e.response.text}") from e
        except Exception as e:
             logging_utility.error("Failed to parse sync API response: %s", str(e))
             raise VectorStoreClientError(f"Invalid response from sync API: {e}") from e


    # Retrieve Store by Collection Async
    async def _retrieve_vector_store_by_collection_async(
        self, collection_name: str
    ) -> ValidationInterface.VectorStoreRead:
        logging_utility.info("Retrieving vector store by collection name %s via API", collection_name)
        response = await self._request_with_retries(
            "GET", "/v1/vector-stores/lookup/collection", params={"name": collection_name}
        )
        return ValidationInterface.VectorStoreRead.model_validate(response)

    # Retrieve Store by Collection Sync
    def retrieve_vector_store_by_collection(
        self, collection_name: str
    ) -> ValidationInterface.VectorStoreRead:
        """Synchronously retrieves vector store metadata by its collection name."""
        return asyncio.run(self._retrieve_vector_store_by_collection_async(collection_name))
