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

from entities_sdk.clients.file_processor import FileProcessor
from entities_sdk.clients.vector_store_manager import VectorStoreManager

load_dotenv()
logging_utility = UtilsInterface.LoggingUtility()


class VectorStoreClientError(Exception):
    """Custom exception for VectorStoreClient errors."""
    pass


class VectorStoreClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 vector_store_host: Optional[str] = 'localhost'):
        self.base_url = base_url or os.getenv("BASE_URL")
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.base_url:
            raise VectorStoreClientError("BASE_URL must be provided either as an argument or in environment variables.")

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # Use AsyncClient for potential async operations like file processing
        self.api_client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=30.0)  # Increased timeout
        self.sync_api_client = httpx.Client(base_url=self.base_url, headers=headers, timeout=30.0)  # For sync methods

        self.vector_store_host = vector_store_host
        self.vector_manager = VectorStoreManager(vector_store_host=self.vector_store_host)
        self.identifier_service = IdentifierService()

        self.file_processor = FileProcessor()

    async def close(self):
        await self.api_client.aclose()
        self.sync_api_client.close()
        # Qdrant client close if necessary (depends on VectorStoreManager implementation)
        # self.vector_manager.get_client().close()

    async def _parse_response(self, response: httpx.Response) -> Any:
        try:
            response.raise_for_status()  # Raise exception for 4xx/5xx errors
            if response.status_code == 204:  # No Content
                return None
            return response.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error("API request failed: Status %d, Response: %s",
                                  e.response.status_code, e.response.text)
            # You could parse the error detail from response.json() if available
            raise VectorStoreClientError(f"API Error: {e.response.status_code} - {e.response.text}") from e
        except Exception as e:
            logging_utility.error("Failed to parse API response: %s", str(e))
            raise VectorStoreClientError(f"Invalid response from API: {response.text}") from e

    async def _request_with_retries(self, method: str, url: str, is_async: bool = True, **kwargs) -> Any:
        retries = 3
        client = self.api_client if is_async else self.sync_api_client
        last_exception = None

        for attempt in range(retries):
            try:
                response = await client.request(method, url, **kwargs) if is_async else client.request(method, url, **kwargs)
                return await self._parse_response(response)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_exception = e
                # Only retry on 5xx errors or network issues
                should_retry = isinstance(e, (httpx.TimeoutException, httpx.NetworkError)) or \
                    (isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500)

                if should_retry and attempt < retries - 1:
                    wait_time = 2 ** attempt
                    logging_utility.warning("Retrying request (attempt %d/%d) to %s %s after %d s. Error: %s",
                                            attempt + 1, retries, method, url, wait_time, str(e))
                    await asyncio.sleep(wait_time)  # Use asyncio.sleep for async client
                    continue
                else:
                    # Log final failure before raising
                    logging_utility.error("API Request failed permanently after %d attempts to %s %s. Last Error: %s",
                                          attempt + 1, method, url, str(e))
                    # Re-raise the caught exception or a custom one
                    if isinstance(e, httpx.HTTPStatusError):
                        raise VectorStoreClientError(f"API Error: {e.response.status_code} - {e.response.text}") from e
                    else:
                        raise VectorStoreClientError(f"API Communication Error: {str(e)}") from e
            except Exception as e:  # Catch other unexpected errors during request/parsing
                logging_utility.error("Unexpected error during API request to %s %s: %s", method, url, str(e))
                raise VectorStoreClientError(f"Unexpected API Client Error: {str(e)}") from e
        # This part should technically be unreachable due to re-raising in the loop
        raise VectorStoreClientError("Request failed after retries.") from last_exception

    async def create_vector_store(
        self, name: str,
        user_id: str,
        vector_size: int = 384,  # Default or fetch from config
        distance_metric: str = "Cosine",  # Use models.Distance.COSINE if using qdrant client models
        config: Optional[Dict[str, Any]] = None
    ) -> ValidationInterface.VectorStoreRead:
        """
        Creates a vector store:
        1. Generates a unique ID.
        2. Creates the collection in Qdrant.
        3. Registers the store metadata via API call.
        """
        shared_id = self.identifier_service.generate_vector_id()  # e.g., vs_...
        collection_name = shared_id  # Use the same ID for Qdrant collection

        logging_utility.info("Attempting to create Qdrant collection '%s'", collection_name)
        try:
            # Ensure distance metric is valid for Qdrant (using string directly here)
            qdrant_success = self.vector_manager.create_store(
                store_name=collection_name,  # Use unique ID for Qdrant
                vector_size=vector_size,
                distance=distance_metric.upper()  # Qdrant expects uppercase usually
            )
            if not qdrant_success:  # Assuming create_store returns bool or raises error
                raise VectorStoreClientError(f"Failed to create Qdrant collection '{collection_name}'")
            logging_utility.info("Successfully created Qdrant collection '%s'", collection_name)
        except Exception as e:
            logging_utility.error("Qdrant collection creation failed for '%s': %s", collection_name, str(e))
            raise VectorStoreClientError(f"Failed to create vector store backend: {str(e)}") from e

        # If Qdrant creation succeeded, register in DB via API
        logging_utility.info("Registering vector store '%s' (ID: %s) via API", name, shared_id)
        db_payload = {
            "shared_id": shared_id,
            "name": name,
            "user_id": user_id,
            "vector_size": vector_size,
            "distance_metric": distance_metric.upper(),
            "config": config or {},
            # collection_name is derived from shared_id in the API service now
        }
        try:
            response_data = await self._request_with_retries("POST", "/v1/vector-stores", json=db_payload)
            logging_utility.info("Successfully registered vector store '%s' via API", name)
            return ValidationInterface.VectorStoreRead.model_validate(response_data)
        except Exception as api_error:
            # Rollback Qdrant creation if API registration fails
            logging_utility.error("API registration failed for store '%s' (ID: %s). Rolling back Qdrant collection. Error: %s",
                                  name, shared_id, str(api_error))
            try:
                self.vector_manager.delete_store(collection_name)
                logging_utility.info("Rolled back Qdrant collection '%s'", collection_name)
            except Exception as rollback_error:
                logging_utility.error("Failed to rollback Qdrant collection '%s': %s",
                                      collection_name, str(rollback_error))
                # Log this critical state failure
            raise api_error  # Re-raise the original API error

    async def add_file_to_vector_store(
        self,
        vector_store_id: str,
        file_path: Union[str, Path],
        chunk_size: int = 512,
        embedding_model_name: str = "paraphrase-MiniLM-L6-v2",  # Make configurable
        user_metadata: Optional[Dict[str, Any]] = None,
        # source_url: Optional[str] = None # Add if needed
    ) -> ValidationInterface.VectorStoreRead:
        """
        Processes a file, uploads chunks to Qdrant, and registers the file via API.
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 0. Get Vector Store details (needed for collection_name)
        # We need the collection_name which is the same as vector_store_id in our setup
        collection_name = vector_store_id
        # Optional: Verify store exists via API first?
        # try:
        #     store_info = await self.retrieve_vector_store(vector_store_id)
        #     collection_name = store_info.collection_name
        # except VectorStoreClientError:
        #     logging_utility.error(f"Vector store {vector_store_id} not found via API.")
        #     raise

        # 1. Process file (chunk, embed)

        logging_utility.info("Processing file: %s", file_path)
        try:
            # Assuming process_file handles reading and embedding
            processed_data = await self.file_processor.process_file(file_path)
            texts = processed_data["chunks"]
            vectors = processed_data["vectors"]
            # Create metadata for each chunk (important: include original file_path)
            base_metadata = user_metadata or {}
            base_metadata["source"] = str(file_path)  # Use file_path for Qdrant filter key 'source'
            base_metadata["file_name"] = file_path.name

            chunk_metadata = [
                {**base_metadata, "chunk_index": i}
                for i in range(len(texts))
            ]
            logging_utility.info("Processed file '%s' into %d chunks.", file_path.name, len(texts))
        except Exception as e:
            logging_utility.error("Failed to process file %s: %s", file_path, str(e))
            raise VectorStoreClientError(f"File processing failed: {str(e)}") from e

        # 2. Upload to Qdrant
        logging_utility.info("Uploading %d chunks for '%s' to Qdrant collection '%s'",
                             len(texts), file_path.name, collection_name)
        try:
            qdrant_result = self.vector_manager.add_to_store(
                store_name=collection_name,  # Use collection_name for Qdrant
                texts=texts,
                vectors=vectors,
                metadata=chunk_metadata  # Ensure metadata includes 'source': file_path
            )
            # Check qdrant_result for success/failure if possible
            logging_utility.info("Successfully uploaded chunks to Qdrant for '%s'. Result: %s",
                                 file_path.name, qdrant_result)
        except Exception as e:
            logging_utility.error("Qdrant upload failed for file %s to collection %s: %s",
                                  file_path.name, collection_name, str(e))
            raise VectorStoreClientError(f"Vector store upload failed: {str(e)}") from e

        # 3. Register file in DB via API
        # Generate a unique ID for the VectorStoreFile record, or use one if available (e.g., from a main Files table)
        file_record_id = f"vsf_{uuid.uuid4()}"
        api_payload = {
            "file_id": file_record_id,
            "file_name": file_path.name,
            "file_path": str(file_path),  # Store the path used in Qdrant metadata['source']
            "status": "completed",  # Assuming success if we reached here
            "meta_data": user_metadata or {}  # Store user metadata if provided
        }
        logging_utility.info("Registering file '%s' (ID: %s) in vector store '%s' via API",
                             file_path.name, file_record_id, vector_store_id)
        try:
            # Correct API endpoint: /v1/vector-stores/{vector_store_id}/files
            response_data = await self._request_with_retries("POST", f"/v1/vector-stores/{vector_store_id}/files", json=api_payload)
            logging_utility.info("Successfully registered file '%s' via API.", file_path.name)
            return ValidationInterface.VectorStoreRead.model_validate(response_data)
        except Exception as api_error:
            # CRITICAL: Qdrant has the data, but DB registration failed.
            # Manual cleanup or retry mechanism might be needed.
            # For now, log profusely and raise.
            logging_utility.critical(
                "QDRANT UPLOAD SUCCEEDED for file '%s' to store '%s', BUT API registration FAILED. "
                "Manual reconciliation may be required. Error: %s",
                file_path.name, vector_store_id, str(api_error)
            )
            # Consider attempting to delete from Qdrant here, but that could also fail.
            # self.delete_file_from_vector_store(vector_store_id, str(file_path)) # This might hide the root cause
            raise api_error

    async def search_vector_store(
        self,
        vector_store_id: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        # Add other search params matching Qdrant/VectorManager if needed
        # score_threshold: float = 0.0,
        # page: int = 1, page_size: int = 10,
        # score_boosts: Optional[Dict[str, float]] = None,
        # search_type: Optional[str] = None,
        # explain: bool = False
    ) -> List[Dict[str, Any]]:  # Return type likely Qdrant search results
        """Performs semantic search directly against Qdrant."""
        # 0. Get Vector Store details (needed for collection_name and vector size/model)
        try:
            # Use sync client for potentially faster lookup if needed before async embedding
            store_info = self.retrieve_vector_store_sync(vector_store_id)
            collection_name = store_info.collection_name
            # You might need vector_size or model info here if embedding happens client-side
        except VectorStoreClientError:
            logging_utility.error(f"Vector store {vector_store_id} not found via API.")
            raise

        # 1. Embed the query text (assuming FileProcessor can embed single strings)
        # Reuse file processor instance or create one

        embedding_model_name = "paraphrase-MiniLM-L6-v2"  # Get from store_info or config

        try:
            query_vector = self.file_processor.embedding_model.encode(query_text).tolist()
        except Exception as e:
            logging_utility.error("Failed to embed query text: %s", str(e))
            raise VectorStoreClientError(f"Query embedding failed: {str(e)}") from e

        # 2. Query Qdrant
        logging_utility.info("Searching Qdrant collection '%s' with top_k=%d", collection_name, top_k)
        try:
            # Pass relevant parameters to vector_manager
            search_results = self.vector_manager.query_store(
                store_name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters  # Pass Qdrant compatible filters if provided
                # Add other params like score_threshold, offset, limit etc.
            )
            logging_utility.info("Qdrant search completed. Found %d results.", len(search_results))
            # No redundant API call needed here. Return Qdrant results directly.
            return search_results
        except Exception as e:
            logging_utility.error("Qdrant search failed for collection %s: %s", collection_name, str(e))
            raise VectorStoreClientError(f"Vector store search failed: {str(e)}") from e

    async def delete_vector_store(self, vector_store_id: str, permanent: bool = False) -> Dict[str, Any]:
        """
        Deletes a vector store:
        1. Deletes the collection from Qdrant.
        2. Deletes/Marks deleted the store metadata via API call.
        """
        # Need collection_name, which is same as vector_store_id
        collection_name = vector_store_id

        # 1. Delete from Qdrant first (idempotent, safe to retry)
        logging_utility.info("Attempting to delete Qdrant collection '%s'", collection_name)
        qdrant_result = None
        try:
            qdrant_result = self.vector_manager.delete_store(collection_name)
            logging_utility.info("Qdrant delete result for collection '%s': %s", collection_name, qdrant_result)
            # Note: delete_store might return True/False or raise error on failure
        except Exception as e:
            # If permanent deletion is requested, failing Qdrant delete is serious.
            # If soft delete, maybe we can proceed with API call? Depends on requirements.
            logging_utility.error("Qdrant collection deletion failed for '%s': %s. Proceeding with API call might leave orphaned data.",
                                  collection_name, str(e))
            if permanent:
                raise VectorStoreClientError(f"Failed to delete vector store backend: {str(e)}") from e
            # If not permanent, log warning and continue to mark deleted in DB

        # 2. Call API to delete/mark deleted in DB
        logging_utility.info("Calling API to %s delete vector store '%s'",
                             "permanently" if permanent else "soft", vector_store_id)
        try:
            # API endpoint: DELETE /v1/vector-stores/{vector_store_id}?permanent={permanent}
            api_response = await self._request_with_retries(
                "DELETE", f"/v1/vector-stores/{vector_store_id}",
                params={"permanent": permanent}
            )
            logging_utility.info("API delete call successful for vector store '%s'.", vector_store_id)
            return {
                "vector_store_id": vector_store_id,
                "status": "deleted",
                "permanent": permanent,
                "qdrant_result": qdrant_result,  # Include Qdrant status if available
                "api_result": api_response  # Include API status if available
            }
        except Exception as api_error:
            logging_utility.error("API delete call failed for vector store '%s'. Qdrant status: %s. Error: %s",
                                  vector_store_id, qdrant_result, str(api_error))
            # If Qdrant delete succeeded but API failed, DB state is inconsistent.
            raise api_error  # Re-raise API error

    async def delete_file_from_vector_store(self, vector_store_id: str, file_path: str) -> Dict[str, Any]:
        """
        Deletes a file's chunks from Qdrant and its metadata record via API.
        Uses file_path to identify chunks in Qdrant and the record in the DB.
        """
        # Need collection_name, which is same as vector_store_id
        collection_name = vector_store_id

        # 1. Delete from Qdrant using filter on metadata['source']
        logging_utility.info("Attempting to delete chunks for file '%s' from Qdrant collection '%s'",
                             file_path, collection_name)
        qdrant_result = None
        try:
            # delete_file_from_store should use a filter like:
            # models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=file_path))])
            qdrant_result = self.vector_manager.delete_file_from_store(collection_name, file_path)
            logging_utility.info("Qdrant delete result for file '%s': %s", file_path, qdrant_result)
        except Exception as e:
            logging_utility.error("Qdrant deletion failed for file '%s' in collection '%s': %s",
                                  file_path, collection_name, str(e))
            # Don't proceed with API delete if Qdrant failed, prevents orphaned DB record
            raise VectorStoreClientError(f"Failed to delete file from vector store backend: {str(e)}") from e

        # 2. Call API to delete the VectorStoreFile record
        logging_utility.info("Calling API to delete record for file '%s' in vector store '%s'",
                             file_path, vector_store_id)
        try:
            # API endpoint: DELETE /v1/vector-stores/{vector_store_id}/files?file_path={file_path}
            # URL encode file_path
            encoded_file_path = httpx.URL(file_path).path  # Basic encoding
            api_response = await self._request_with_retries(
                "DELETE", f"/v1/vector-stores/{vector_store_id}/files",
                params={"file_path": encoded_file_path}  # Pass file_path as query param
            )
            logging_utility.info("API delete call successful for file record '%s'.", file_path)
            return {
                "vector_store_id": vector_store_id,
                "file_path": file_path,
                "status": "deleted",
                "qdrant_result": qdrant_result,
                "api_result": api_response
            }
        except Exception as api_error:
            # CRITICAL: Qdrant delete succeeded, but API failed. DB file count is now wrong.
            logging_utility.critical(
                "QDRANT DELETE SUCCEEDED for file '%s' in store '%s', BUT API deletion FAILED. "
                "DB file count is likely incorrect. Manual reconciliation may be required. Error: %s",
                file_path, vector_store_id, str(api_error)
            )
            raise api_error

    async def list_store_files(self, vector_store_id: str) -> List[ValidationInterface.VectorStoreRead]:
        """Lists files associated with a vector store by querying the API (DB)."""
        logging_utility.info("Listing files for vector store '%s' via API", vector_store_id)
        try:

            response_data = await self._request_with_retries("GET", f"/v1/vector-stores/{vector_store_id}/files")
            # Assuming API returns a list of file objects conforming to VectorStoreFileRead
            return [ValidationInterface.VectorStoreRead.model_validate(item) for item in response_data]
        except Exception as api_error:
            logging_utility.error("Failed to list files for store '%s' via API: %s", vector_store_id, str(api_error))
            raise api_error

    # --- Assistant & User Methods ---

    async def attach_vector_store_to_assistant(self, vector_store_id: str, assistant_id: str) -> bool:
        logging_utility.info("Attaching vector store %s to assistant %s via API", vector_store_id, assistant_id)

        response = await self._request_with_retries("POST", f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/attach")
        return bool(response)

    async def detach_vector_store_from_assistant(self, vector_store_id: str, assistant_id: str) -> bool:
        logging_utility.info("Detaching vector store %s from assistant %s via API", vector_store_id, assistant_id)

        response = await self._request_with_retries("DELETE", f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/detach")
        return bool(response)

    async def get_vector_stores_for_assistant(self, assistant_id: str) -> List[ValidationInterface.VectorStoreRead]:
        logging_utility.info("Getting vector stores for assistant %s via API", assistant_id)
        response = await self._request_with_retries("GET", f"/v1/assistants/{assistant_id}/vector-stores")
        return [ValidationInterface.VectorStoreRead.model_validate(item) for item in response]

    async def get_stores_by_user(self, user_id: str) -> List[ValidationInterface.VectorStoreRead]:
        logging_utility.info("Getting vector stores for user %s via API", user_id)
        response = await self._request_with_retries("GET", f"/v1/users/{user_id}/vector-stores")
        return [ValidationInterface.VectorStoreRead.model_validate(item) for item in response]

    async def retrieve_vector_store(self, vector_store_id: str) -> ValidationInterface.VectorStoreRead:
        """Retrieves vector store metadata by its ID via API."""
        logging_utility.info("Retrieving vector store %s via API", vector_store_id)

        response = await self._request_with_retries("GET", f"/v1/vector-stores/{vector_store_id}")
        return ValidationInterface.VectorStoreRead.model_validate(response)

    def retrieve_vector_store_sync(self, vector_store_id: str) -> ValidationInterface.VectorStoreRead:
        """Synchronous version of retrieve_vector_store."""
        logging_utility.info("Retrieving vector store %s via sync API", vector_store_id)
        response = self.sync_api_client.get(f"{self.base_url}/v1/vector-stores/{vector_store_id}")
        response.raise_for_status()  # Handle errors
        return ValidationInterface.VectorStoreRead.model_validate(response.json())

    async def retrieve_vector_store_by_collection(self, collection_name: str) -> ValidationInterface.VectorStoreRead:
        """Retrieves vector store metadata by its collection name via API."""
        logging_utility.info("Retrieving vector store by collection name %s via API", collection_name)

        response = await self._request_with_retries("GET", f"/v1/vector-stores/lookup/collection", params={"name": collection_name})
        return ValidationInterface.VectorStoreRead.model_validate(response)
