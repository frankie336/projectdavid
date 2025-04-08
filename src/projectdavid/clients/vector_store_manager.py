import time
import uuid
from typing import Dict, List, Optional

from dotenv import load_dotenv
from projectdavid_common import UtilsInterface
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .base_vector_store import (
    BaseVectorStore,
    StoreExistsError,
    StoreNotFoundError,
    VectorStoreError,
)

load_dotenv()
logging_utility = UtilsInterface.LoggingUtility()


class VectorStoreManager(BaseVectorStore):

    def __init__(self, vector_store_host: str = "localhost", port: int = 6333):

        self.vector_store_host = vector_store_host
        self.client = QdrantClient(host=self.vector_store_host, port=port)
        self.active_stores: Dict[str, dict] = {}
        logging_utility.info(
            f"Initialized HTTP-based VectorStoreManager. Source: {__file__}"
        )

    def _generate_vector_id(self) -> str:
        return str(uuid.uuid4())

        # Inside class VectorStoreManager:

    def create_store(
        self,
        collection_name: str,  # <<< Changed first argument to 'collection_name'
        vector_size: int = 384,
        distance: str = "COSINE",
        # Removed the unused 'store_name' parameter from signature
    ) -> dict:  # Return type still dict for status

        # Check if a collection with this name *already* exists in Qdrant (more robust check)
        try:
            # Optional: Check existing collections instead of relying only on internal dict
            existing_collections = self.client.get_collections().collections
            if any(c.name == collection_name for c in existing_collections):
                # Or handle differently, maybe return info? For now, raise.
                raise StoreExistsError(
                    f"Qdrant collection '{collection_name}' already exists."
                )
        except Exception as e:
            # Handle potential connection errors during check
            logging_utility.warning(f"Could not check existing collections: {e}")
            # Decide if you want to proceed cautiously or raise

        # Use the unique collection_name provided by the client for Qdrant interaction
        logging_utility.info(
            f"Attempting to create Qdrant collection: {collection_name}"
        )
        try:
            normalized_distance = distance.upper()
            if normalized_distance not in Distance.__members__:
                raise ValueError(f"Invalid distance metric '{distance}'.")

            # *** Use collection_name parameter here ***
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance[normalized_distance]
                ),
            )

            # Keep internal track using the collection_name (unique ID) as the key
            # If you need to store the user-friendly 'name', the client/API layer handles that.
            self.active_stores[collection_name] = {  # <<< Use collection_name as key
                "created_at": int(time.time()),
                "vector_size": vector_size,
                "distance": normalized_distance,
            }
            logging_utility.info(
                f"Successfully created Qdrant collection: {collection_name}"
            )
            # Return info about the collection that was created
            return {"collection_name": collection_name, "status": "created"}

        except Exception as e:
            logging_utility.error(
                f"Create store failed for collection '{collection_name}': {str(e)}"
            )
            # Raise a specific error type if possible
            raise VectorStoreError(
                f"Qdrant collection creation failed: {str(e)}"
            ) from e

    def add_to_store(
        self,
        store_name: str,
        texts: List[str],
        vectors: List[List[float]],
        metadata: List[dict],
    ):
        if not vectors:
            raise ValueError("Empty vectors list")
        expected_size = len(vectors[0])
        for i, vec in enumerate(vectors):
            if len(vec) != expected_size:
                raise ValueError(f"Vector {i} size mismatch")
            if not all(isinstance(v, float) for v in vec):
                raise TypeError(f"Vector {i} contains non-floats")
        points = [
            PointStruct(
                id=self._generate_vector_id(),
                vector=vec,
                payload={"text": txt, "metadata": meta},
            )
            for txt, vec, meta in zip(texts, vectors, metadata)
        ]
        try:
            self.client.upsert(collection_name=store_name, points=points)
            return {"status": "success", "points_inserted": len(points)}
        except Exception as e:
            logging_utility.error(f"Add to store failed: {str(e)}")
            raise VectorStoreError(f"Insertion failed: {str(e)}")

    def query_store(
        self,
        store_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
        score_threshold: float = 0.0,
        offset: int = 0,
        limit: Optional[int] = None,
        score_boosts: Optional[Dict[str, float]] = None,
        search_type: Optional[str] = None,
        explain: bool = False,
    ) -> List[dict]:
        # Default limit to top_k if not provided.
        if limit is None:
            limit = top_k
        try:
            flt = None
            if filters and "key" in filters and "value" in filters:
                flt = Filter(
                    must=[
                        FieldCondition(
                            key=filters["key"], match=MatchValue(value=filters["value"])
                        )
                    ]
                )
            # Prepare extra parameters for the search request.
            extra_params = {}
            if score_boosts is not None:
                extra_params["score_boosts"] = score_boosts
            if search_type is not None:
                extra_params["search_type"] = search_type
            extra_params["explain"] = explain

            results = self.client.search(
                collection_name=store_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                query_filter=flt,
                **extra_params,
            )
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "text": r.payload.get("text"),
                    "metadata": r.payload.get("metadata", {}),
                }
                for r in results
            ]
        except Exception as e:
            logging_utility.error(f"Query failed: {str(e)}")
            raise VectorStoreError(f"Query failed: {str(e)}")

    def delete_store(self, store_name: str) -> dict:
        if store_name not in self.active_stores:
            raise StoreNotFoundError(f"Store {store_name} not found")
        try:
            self.client.delete_collection(collection_name=store_name)
            del self.active_stores[store_name]
            return {"name": store_name, "status": "deleted"}
        except Exception as e:
            logging_utility.error(f"Delete failed: {str(e)}")
            raise VectorStoreError(f"Store deletion failed: {str(e)}")

    def get_store_info(self, store_name: str) -> dict:
        if store_name not in self.active_stores:
            raise StoreNotFoundError(f"Store {store_name} not found")
        try:
            info = self.client.get_collection(collection_name=store_name)
            return {
                "name": store_name,
                "status": "active",
                "vectors_count": info.points_count,
                "configuration": info.config.params["default"],
                "created_at": self.active_stores[store_name]["created_at"],
            }
        except Exception as e:
            logging_utility.error(f"Store info failed: {str(e)}")
            raise VectorStoreError(f"Info retrieval failed: {str(e)}")

    def delete_file_from_store(self, store_name: str, file_path: str) -> dict:
        try:
            payload = {
                "filter": {
                    "must": [{"key": "metadata.source", "match": {"value": file_path}}]
                }
            }
            res = self.client.post(
                f"/collections/{store_name}/points/delete", json=payload
            )
            res.raise_for_status()
            return {
                "deleted_file": file_path,
                "store_name": store_name,
                "status": "success",
            }
        except Exception as e:
            logging_utility.error(f"File deletion failed: {str(e)}")
            raise VectorStoreError(f"File deletion failed: {str(e)}")

    def get_point_by_id(self, store_name: str, point_id: str) -> dict:
        """
        Retrieve a specific point by its ID from the given collection.

        Parameters:
            store_name (str): The name of the Qdrant collection.
            point_id (str): The unique identifier of the point.

        Returns:
            dict: A dictionary containing the point data (e.g., id, payload, vector).

        Raises:
            VectorStoreError: If the point is not found or if the request fails.
        """
        try:
            # Use Qdrant's retrieve method, which accepts a list of IDs.
            result = self.client.retrieve(collection_name=store_name, ids=[point_id])

            # Qdrant returns a dict with a "result" key containing a list of points.
            points = result.get("result", [])
            if not points:
                raise VectorStoreError(
                    f"Point '{point_id}' not found in store '{store_name}'."
                )
            return points[0]
        except Exception as e:
            logging_utility.error(f"Get point by ID failed: {str(e)}")
            raise VectorStoreError(f"Fetch failed: {str(e)}")

    def list_store_files(self, store_name: str) -> List[str]:
        try:
            scroll_url = f"/collections/{store_name}/points/scroll"
            payload = {"limit": 100, "with_payload": ["metadata.source"]}
            seen = set()
            while True:
                res = self.client.http.post(scroll_url, json=payload)
                res.raise_for_status()
                result = res.json()["result"]
                for point in result["points"]:
                    source = point.get("payload", {}).get("metadata", {}).get("source")
                    if source:
                        seen.add(source)
                if not result.get("next_page_offset"):
                    break
                payload["offset"] = result["next_page_offset"]
            return sorted(seen)
        except Exception as e:
            logging_utility.error(f"List store files failed: {str(e)}")
            raise VectorStoreError(f"List files failed: {str(e)}")

    def get_client(self):
        return self.client

    def health_check(self) -> bool:
        try:
            collections = self.client.get_collections()
            return bool(collections)
        except Exception:
            return False
