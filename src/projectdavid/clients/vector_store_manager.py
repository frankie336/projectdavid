"""
projectdavid.clients.vector_store_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Light wrapper around *qdrant‑client* that hides collection / filter
details from the higher‑level SDK.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Union

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
        store_name: str,
        vector_size: int = 384,
        distance: str = "COSINE",
        vectors_config: Optional[Dict[str, qdrant.VectorParams]] = None,
    ) -> dict:
        """
        Create or recreate a Qdrant collection.

        • If *vectors_config* is provided → use it verbatim (multi-vector schema).
        • Otherwise create a classic single-vector collection *without* naming the
          vector field – so upserts can omit ``vector_name``.
        """
        try:
            # ── pre-existence check ────────────────────────────────────────────
            if any(
                col.name == store_name
                for col in self.client.get_collections().collections
            ):
                raise StoreExistsError(f"Collection '{store_name}' already exists")

            dist = distance.upper()
            if dist not in qdrant.Distance.__members__:
                raise ValueError(f"Invalid distance metric '{distance}'")

            # ── choose schema ──────────────────────────────────────────────────
            config: Union[qdrant.VectorParams, Dict[str, qdrant.VectorParams]]
            if vectors_config:  # caller supplied full mapping
                config = vectors_config  # e.g. {"text_vec": ..., "img_vec": ...}
            else:  # default = single unnamed vector
                config = qdrant.VectorParams(
                    size=vector_size,
                    distance=qdrant.Distance[dist],
                )

            # ── (re)create collection ─────────────────────────────────────────
            self.client.recreate_collection(
                collection_name=store_name,
                vectors_config=config,
            )

            # ── bookkeeping ───────────────────────────────────────────────────
            if isinstance(config, dict):
                fields: List[Optional[str]] = list(config.keys())
            else:  # unnamed field — no vector field name to record
                fields = []

            self.active_stores[store_name] = {
                "created_at": int(time.time()),
                "vector_size": vector_size,
                "distance": dist,
                "fields": fields,
            }
            log.info("Created Qdrant collection %s", store_name)
            return {"collection_name": store_name, "status": "created"}

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
                "configuration": info.config.params,
                "created_at": self.active_stores[store_name]["created_at"],
                "fields": self.active_stores[store_name].get("fields"),
            }
        except Exception as e:
            log.error("Store info failed: %s", e)
            raise VectorStoreError(f"Info retrieval failed: {e}") from e

    def add_to_store(
        self,
        store_name: str,
        texts: List[str],
        vectors: List[List[float]],
        metadata: List[dict],
        vector_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upsert vectors + payloads into *store_name*.

        If *vector_name* is omitted the manager:

        • auto-detects the single vector field for classic (unnamed) collections
        • auto-detects the sole key for named-vector collections with exactly one field
        • raises if multiple named fields exist.
        """

        # ─── input validation ───────────────────────────────────────────────
        if not vectors:
            raise ValueError("Empty vectors list")
        expected = len(vectors[0])
        for i, vec in enumerate(vectors):
            if len(vec) != expected or not all(isinstance(v, float) for v in vec):
                raise ValueError(f"Vector {i} malformed: expected {expected} floats")

        # ─── auto-detect vector field ───────────────────────────────────────
        if vector_name is None:
            coll_info = self.client.get_collection(collection_name=store_name)
            v_cfg = coll_info.config.params.vectors

            if isinstance(v_cfg, dict):  # modern named-vector schema
                vector_fields = list(v_cfg.keys())
                if len(vector_fields) == 1:  # exactly one → safe default
                    vector_name = vector_fields[0]
                    log.debug(
                        "Auto-detected vector_name=%r for store=%s",
                        vector_name,
                        store_name,
                    )
                else:  # >1 named fields → ambiguous
                    raise ValueError(
                        f"Multiple vector fields {vector_fields}; please specify vector_name"
                    )
            else:
                # legacy single-vector schema → leave vector_name as None
                log.debug(
                    "Collection %s uses legacy single-vector schema; "
                    "upserting without vector_name",
                    store_name,
                )

        # ─── build points payload ───────────────────────────────────────────
        points = [
            qdrant.PointStruct(
                id=self._generate_vector_id(),
                vector=vec,
                payload={"text": txt, **meta},
            )
            for txt, vec, meta in zip(texts, vectors, metadata)
        ]

        # ─── upsert with backward-compat for old qdrant-client builds ───────
        import inspect  # keep local to avoid top-level dependency if absent elsewhere

        upsert_sig = inspect.signature(self.client.upsert)
        supports_vector_name = "vector_name" in upsert_sig.parameters

        upsert_kwargs: Dict[str, Any] = {
            "collection_name": store_name,
            "points": points,
            "wait": True,
        }
        if supports_vector_name and vector_name is not None:
            upsert_kwargs["vector_name"] = vector_name

        try:
            self.client.upsert(**upsert_kwargs)
            return {"status": "success", "points_inserted": len(points)}
        except Exception as exc:  # noqa: BLE001
            log.error("Add-to-store failed: %s", exc, exc_info=True)
            raise VectorStoreError(f"Insertion failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # search / query
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dict_to_filter(filters: dict) -> Filter:
        """
        Converts a nested filter dict into a Qdrant-compatible Filter object.
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
        *,
        vector_field: Optional[str] = None,
        score_threshold: float = 0.0,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """
        Run a similarity search against *store_name*.

        • Handles both modern Qdrant (v1.10+ using query_points) and legacy clients (using search).
        • `vector_field` lets you target a non-default vector column.
        """

        limit = limit or top_k
        flt = self._dict_to_filter(filters) if filters else None

        try:
            # ── 1. Try modern Qdrant v1.10+ API (query_points) ──
            response = self.client.query_points(
                collection_name=store_name,
                query=query_vector,
                query_filter=flt,
                limit=limit,
                offset=offset,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            res = response.points

        except AttributeError:
            # ── 2. Fallback for older Qdrant clients ──
            try:
                res = self.client.search(
                    collection_name=store_name,
                    query_vector=query_vector,
                    filter=flt,
                    limit=limit,
                    offset=offset,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
            except TypeError:
                res = self.client.search(
                    collection_name=store_name,
                    query_vector=query_vector,
                    query_filter=flt,
                    limit=limit,
                    offset=offset,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
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

    def get_point_by_id(self, store_name: str, point_id: str) -> Dict[str, Any]:
        try:
            res = self.client.retrieve(collection_name=store_name, ids=[point_id])
            pts = res.get("result") if isinstance(res, dict) else res
            if not pts:
                raise VectorStoreError(f"Point '{point_id}' not found")
            pt = pts[0]
            return {
                "id": pt.id,
                "payload": pt.payload if pt.payload is not None else {},
                "vector": pt.vector,
            }
        except Exception as e:
            log.error("Get point failed: %s", e)
            raise VectorStoreError(f"Fetch failed: {e}") from e

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def get_client(self) -> QdrantClient:
        return self.client
