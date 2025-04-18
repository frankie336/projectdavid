"""
projectdavid.clients.vector_store_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Light wrapper around *qdrant‑client* that hides collection / filter
details from the higher‑level SDK.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from projectdavid_common import UtilsInterface
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant  # unified import
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from .base_vector_store import (
    BaseVectorStore,
    StoreExistsError,
    StoreNotFoundError,
    VectorStoreError,
)

load_dotenv()
log = UtilsInterface.LoggingUtility()


class VectorStoreManager(BaseVectorStore):
    # ------------------------------------------------------------------ #
    # lifecycle helpers
    # ------------------------------------------------------------------ #
    def __init__(self, vector_store_host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=vector_store_host, port=port)
        self.active_stores: Dict[str, dict] = {}
        log.info(
            "Initialized HTTP‑based VectorStoreManager (host=%s)", vector_store_host
        )

    @staticmethod
    def _generate_vector_id() -> str:
        return str(uuid.uuid4())

    # ------------------------------------------------------------------ #
    # collection management
    # ------------------------------------------------------------------ #
    def create_store(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance: str = "COSINE",
    ) -> dict:
        try:
            # quick existence check
            if any(
                col.name == collection_name
                for col in self.client.get_collections().collections
            ):
                raise StoreExistsError(f"Collection '{collection_name}' already exists")

            dist = distance.upper()
            if dist not in qdrant.Distance.__members__:
                raise ValueError(f"Invalid distance metric '{distance}'")

            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=qdrant.VectorParams(
                    size=vector_size, distance=qdrant.Distance[dist]
                ),
            )
            self.active_stores[collection_name] = {
                "created_at": int(time.time()),
                "vector_size": vector_size,
                "distance": dist,
            }
            log.info("Created Qdrant collection %s", collection_name)
            return {"collection_name": collection_name, "status": "created"}

        except Exception as e:
            log.error("Create store failed: %s", e)
            raise VectorStoreError(f"Qdrant collection creation failed: {e}") from e

    def delete_store(self, store_name: str) -> dict:
        if store_name not in self.active_stores:
            raise StoreNotFoundError(store_name)
        try:
            self.client.delete_collection(collection_name=store_name)
            del self.active_stores[store_name]
            return {"name": store_name, "status": "deleted"}
        except Exception as e:
            log.error("Delete failed: %s", e)
            raise VectorStoreError(f"Store deletion failed: {e}") from e

    def get_store_info(self, store_name: str) -> dict:
        if store_name not in self.active_stores:
            raise StoreNotFoundError(store_name)
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
            log.error("Store info failed: %s", e)
            raise VectorStoreError(f"Info retrieval failed: {e}") from e

    # ------------------------------------------------------------------ #
    # ingestion helpers
    # ------------------------------------------------------------------ #
    def add_to_store(
        self,
        store_name: str,
        texts: List[str],
        vectors: List[List[float]],
        metadata: List[dict],
    ):
        if not vectors:
            raise ValueError("Empty vectors list")
        expected = len(vectors[0])
        for i, vec in enumerate(vectors):
            if len(vec) != expected or not all(isinstance(v, float) for v in vec):
                raise ValueError(f"Vector {i} malformed")

        points = [
            qdrant.PointStruct(
                id=self._generate_vector_id(),
                vector=vec,
                payload={"text": txt, **meta},
            )
            for txt, vec, meta in zip(texts, vectors, metadata)
        ]
        try:
            self.client.upsert(collection_name=store_name, points=points, wait=True)
            return {"status": "success", "points_inserted": len(points)}
        except Exception as e:
            log.error("Add‑to‑store failed: %s", e)
            raise VectorStoreError(f"Insertion failed: {e}") from e

    # ------------------------------------------------------------------ #
    # search / query
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dict_to_filter(filters: dict) -> Filter:
        """
        Converts a nested filter dict into a Qdrant-compatible Filter object.

        Example input:
        {
            "must": [
                {"key": "metadata.file_name", "match": {"value": "doc.txt"}},
                {"key": "metadata.chunk_index", "range": {"gte": 10, "lte": 50}}
            ],
            "must_not": [
                {"key": "metadata.chunk_index", "range": {"gte": 90}}
            ],
            "should": [
                {"key": "metadata.author", "match": {"value": "Lord Atkin"}}
            ]
        }
        """

        def parse_condition(cond: dict) -> FieldCondition:
            if "key" not in cond:
                raise ValueError("Each condition must have a 'key'")
            if "match" in cond:
                return FieldCondition(
                    key=cond["key"], match=MatchValue(**cond["match"])
                )
            elif "range" in cond:
                return FieldCondition(key=cond["key"], range=Range(**cond["range"]))
            else:
                raise ValueError(f"Unsupported condition format: {cond}")

        return Filter(
            must=[parse_condition(c) for c in filters.get("must", [])] or None,
            must_not=[parse_condition(c) for c in filters.get("must_not", [])] or None,
            should=[parse_condition(c) for c in filters.get("should", [])] or None,
        )

    def query_store(
        self,
        store_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
        score_threshold: float = 0.0,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Run a similarity search that works with any 1.x qdrant‑client."""

        limit = limit or top_k
        flt = self._dict_to_filter(filters) if filters else None

        common: Dict[str, Any] = dict(
            collection_name=store_name,
            query_vector=query_vector,
            limit=limit,
            offset=offset,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )

        try:
            # Newer clients (≥ 1.6) use `filter=`
            res = self.client.search(**common, filter=flt)  # type: ignore[arg-type]
        except AssertionError as ae:
            if "Unknown arguments" not in str(ae):
                raise
            # Older clients use `query_filter=`
            res = self.client.search(**common, query_filter=flt)  # type: ignore[arg-type]

        except Exception as e:
            log.error("Query failed: %s", e)
            raise VectorStoreError(f"Query failed: {e}") from e

        return [
            {
                "id": p.id,
                "score": p.score,
                "text": p.payload.get("text"),
                "metadata": {k: v for k, v in p.payload.items() if k != "text"},
            }
            for p in res
        ]

    # ------------------------------------------------------------------ #
    # point / file deletion helpers
    # ------------------------------------------------------------------ #
    def delete_file_from_store(self, store_name: str, file_path: str) -> dict:
        try:
            cond = qdrant.FieldCondition(
                key="file_path", match=qdrant.MatchValue(value=file_path)
            )
            self.client.delete(
                collection_name=store_name,
                points_selector=qdrant.FilterSelector(
                    filter=qdrant.Filter(must=[cond])
                ),
                wait=True,
            )
            return {
                "deleted_file": file_path,
                "store_name": store_name,
                "status": "success",
            }
        except Exception as e:
            log.error("File deletion failed: %s", e)
            raise VectorStoreError(f"File deletion failed: {e}") from e

    # ------------------------------------------------------------------ #
    # misc helpers
    # ------------------------------------------------------------------ #
    def list_store_files(self, store_name: str) -> List[str]:
        """Return distinct `file_path` payload values present in the collection."""
        try:
            seen = set()
            scroll = self.client.scroll(
                collection_name=store_name,
                with_payload=["file_path"],
                limit=100,
            )
            while scroll[1] is not None:
                for pt in scroll[0]:
                    if fp := pt.payload.get("file_path"):
                        seen.add(fp)
                scroll = self.client.scroll(
                    collection_name=store_name,
                    with_payload=["file_path"],
                    limit=100,
                    offset=scroll[1],
                )
            return sorted(seen)
        except Exception as e:
            log.error("List store files failed: %s", e)
            raise VectorStoreError(f"List files failed: {e}") from e

    def get_point_by_id(self, store_name: str, point_id: str) -> dict:
        try:
            res = self.client.retrieve(collection_name=store_name, ids=[point_id])
            pts = res.get("result") if isinstance(res, dict) else res
            if not pts:
                raise VectorStoreError(f"Point '{point_id}' not found")
            return pts[0]
        except Exception as e:
            log.error("Get point failed: %s", e)
            raise VectorStoreError(f"Fetch failed: {e}") from e

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    # expose raw client if needed
    def get_client(self) -> QdrantClient:
        return self.client
