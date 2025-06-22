"""projectdavid.clients.vector_store_client
---------------------------------------

Token-scoped HTTP client + local Qdrant helper for vector-store operations.
"""

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
from projectdavid.synthesis import reranker, retriever
from projectdavid.synthesis.llm_synthesizer import synthesize_envelope
from projectdavid.utils.vector_search_formatter import make_envelope

load_dotenv()
log = UtilsInterface.LoggingUtility()


def summarize_hits(query: str, hits: List[Dict[str, Any]]) -> str:
    lines = [f"• {h['meta_data']['file_name']} (score {h['score']:.2f})" for h in hits]
    return f"Top files for **{query}**:\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Exceptions
# --------------------------------------------------------------------------- #
class VectorStoreClientError(Exception):
    """Raised on any client-side or API error."""


# --------------------------------------------------------------------------- #
#  Helper schema
# --------------------------------------------------------------------------- #
class VectorStoreFileUpdateStatusInput(BaseModel):
    status: ValidationInterface.StatusEnum = Field(
        ..., description="The new status for the file record."
    )
    error_message: Optional[str] = Field(
        None, description="Error message if status is 'failed'."
    )


# --------------------------------------------------------------------------- #
#  Main client
# --------------------------------------------------------------------------- #
class VectorStoreClient:
    """
    Thin HTTP+Qdrant wrapper.

    • All API requests scoped by X-API-Key.
    • create_vector_store() no longer takes user_id; ownership from token.
    """

    # ------------------------------------------------------------------ #
    #  Construction / cleanup
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        vector_store_host: str = "localhost",
        file_processor_kwargs: Optional[dict] = None,  # 🔶 add arg
    ):
        self.base_url = (base_url or os.getenv("BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.base_url:
            raise VectorStoreClientError("BASE_URL is required.")

        self._base_headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            self._base_headers["X-API-Key"] = self.api_key
        else:
            log.warning("No API key — protected routes will fail.")

        self._sync_api_client = httpx.Client(
            base_url=self.base_url, headers=self._base_headers, timeout=30.0
        )

        # Local helpers ---------------------------------------------------
        self.vector_manager = VectorStoreManager(vector_store_host=vector_store_host)
        self.identifier_service = UtilsInterface.IdentifierService()

        # 🔶 forward kwargs into the upgraded FileProcessor
        # self.file_processor = FileProcessor(**(file_processor_kwargs or {}))
        # Using Stripped down version for now until we move forward with multi-modal stores
        self.file_processor = FileProcessor()

        log.info("VectorStoreClient → %s", self.base_url)

    # Context support ------------------------------------------------------ #
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        await self.aclose()

    # Cleanup -------------------------------------------------------------- #
    async def aclose(self):
        await asyncio.to_thread(self._sync_api_client.close)

    def close(self):
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                warnings.warn(
                    "close() inside running loop — use `await aclose()`",
                    RuntimeWarning,
                )
                self._sync_api_client.close()
                return
        except RuntimeError:
            pass
        asyncio.run(self.aclose())

    # Low-level HTTP helpers ---------------------------------------------- #
    async def _parse_response(self, resp: httpx.Response) -> Any:
        try:
            resp.raise_for_status()
            return None if resp.status_code == 204 else resp.json()
        except httpx.HTTPStatusError as exc:
            log.error("API %d – %s", exc.response.status_code, exc.response.text)
            raise VectorStoreClientError(
                f"API {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise VectorStoreClientError(f"Invalid response: {resp.text}") from exc

    async def _request(self, method: str, url: str, **kwargs) -> Any:
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=self._base_headers,
                    timeout=30.0,
                ) as client:
                    resp = await client.request(method, url, **kwargs)
                    return await self._parse_response(resp)
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
            ) as exc:
                retryable = isinstance(
                    exc, (httpx.TimeoutException, httpx.NetworkError)
                ) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code >= 500
                )
                if retryable and attempt < retries:
                    backoff = 2 ** (attempt - 1)
                    log.warning(
                        "Retry %d/%d %s %s in %ds – %s",
                        attempt,
                        retries,
                        method,
                        url,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise VectorStoreClientError(str(exc)) from exc
        raise VectorStoreClientError("Request failed after retries")

    # Internal async ops -------------------------------------------------- #
    async def _create_vs_async(
        self,
        name: str,
        vector_size: int,
        distance_metric: str,
        config: Optional[Dict[str, Any]],
    ) -> ValidationInterface.VectorStoreRead:
        shared_id = self.identifier_service.generate_vector_id()
        self.vector_manager.create_store(
            collection_name=shared_id,
            vector_size=vector_size,
            distance=distance_metric.upper(),
        )

        payload = {
            "shared_id": shared_id,
            "name": name,
            "vector_size": vector_size,
            "distance_metric": distance_metric.upper(),
            "config": config or {},
        }
        resp = await self._request("POST", "/v1/vector-stores", json=payload)
        return ValidationInterface.VectorStoreRead.model_validate(resp)

    async def _list_my_vs_async(self) -> List[ValidationInterface.VectorStoreRead]:
        resp = await self._request("GET", "/v1/vector-stores")
        return [ValidationInterface.VectorStoreRead.model_validate(r) for r in resp]

    # ------------------------------------------------------------------ #
    # NEW  admin‑aware creation helper
    # ------------------------------------------------------------------ #
    async def _create_vs_for_user_async(
        self,
        owner_id: str,
        name: str,
        vector_size: int,
        distance_metric: str,
        config: Optional[Dict[str, Any]],
    ) -> ValidationInterface.VectorStoreRead:
        shared_id = self.identifier_service.generate_vector_id()
        self.vector_manager.create_store(
            collection_name=shared_id,
            vector_size=vector_size,
            distance=distance_metric.upper(),
        )
        payload = {
            "shared_id": shared_id,
            "name": name,
            "vector_size": vector_size,
            "distance_metric": distance_metric.upper(),
            "config": config or {},
        }
        # pass owner_id as query‑param (backend enforces admin‑only)
        resp = await self._request(
            "POST",
            "/v1/vector-stores",
            json=payload,
            params={"owner_id": owner_id},
        )
        return ValidationInterface.VectorStoreRead.model_validate(resp)

    async def _add_file_async(
        self, vector_store_id: str, p: Path, meta: Optional[Dict[str, Any]]
    ) -> ValidationInterface.VectorStoreFileRead:
        processed = await self.file_processor.process_file(p)
        texts, vectors = processed["chunks"], processed["vectors"]
        line_data = processed.get("line_data") or []

        base_md = (meta or {}) | {"source": str(p), "file_name": p.name}
        file_record_id = f"vsf_{uuid.uuid4()}"

        chunk_md = []
        for i, txt in enumerate(texts):
            payload = {**base_md, "chunk_index": i, "file_id": file_record_id}
            if i < len(line_data):
                payload.update(line_data[i])  # {'page':…, 'lines':…}
            chunk_md.append(payload)

        # 🔑 1. look up the backend store to get its *collection* name
        store = self.retrieve_vector_store_sync(vector_store_id)
        collection_name = store.collection_name

        # 🔑 2. upsert via VectorStoreManager (auto-detects vector field)
        self.vector_manager.add_to_store(
            store_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadata=chunk_md,
        )

        # 3. register the file with the API
        resp = await self._request(
            "POST",
            f"/v1/vector-stores/{vector_store_id}/files",
            json={
                "file_id": file_record_id,
                "file_name": p.name,
                "file_path": str(p),
                "status": "completed",
                "meta_data": meta or {},
            },
        )
        return ValidationInterface.VectorStoreFileRead.model_validate(resp)

    async def _search_vs_async(
        self,
        vector_store_id: str,
        query_text: str,
        top_k: int,
        filters: Optional[Dict] = None,
        vector_store_host: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        # pick local vs. override host
        vector_manager = (
            VectorStoreManager(vector_store_host=vector_store_host)
            if vector_store_host
            else self.vector_manager
        )

        store = self.retrieve_vector_store_sync(vector_store_id)

        # 🔶 choose encoder by vector_size
        if store.vector_size == 1024:  # images collection
            vec = self.file_processor.encode_clip_text(query_text).tolist()
            vector_field = "caption_vector"  # field name in Qdrant
        else:  # 384-D text collection
            vec = self.file_processor.encode_text(query_text).tolist()
            vector_field = None  # default field

        return vector_manager.query_store(
            store_name=store.collection_name,
            query_vector=vec,
            top_k=top_k,
            filters=filters,
            vector_field=vector_field,
        )

    async def _delete_vs_async(self, vector_store_id: str, permanent: bool):
        # collection deletion must use the *collection* name
        store = self.retrieve_vector_store_sync(vector_store_id)
        qres = self.vector_manager.delete_store(store.collection_name)
        await self._request(
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

    async def _delete_file_async(self, vector_store_id: str, file_path: str):
        store = self.retrieve_vector_store_sync(vector_store_id)
        fres = self.vector_manager.delete_file_from_store(
            store.collection_name, file_path
        )
        await self._request(
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

    async def _list_store_files_async(
        self, vector_store_id: str
    ) -> List[ValidationInterface.VectorStoreFileRead]:
        resp = await self._request("GET", f"/v1/vector-stores/{vector_store_id}/files")
        return [
            ValidationInterface.VectorStoreFileRead.model_validate(item)
            for item in resp
        ]

    async def _update_file_status_async(
        self,
        vector_store_id: str,
        file_id: str,
        status: ValidationInterface.StatusEnum,
        error_message: Optional[str] = None,
    ) -> ValidationInterface.VectorStoreFileRead:
        payload = VectorStoreFileUpdateStatusInput(
            status=status, error_message=error_message
        ).model_dump(exclude_none=True)
        resp = await self._request(
            "PATCH",
            f"/v1/vector-stores/{vector_store_id}/files/{file_id}",
            json=payload,
        )
        return ValidationInterface.VectorStoreFileRead.model_validate(resp)

    async def _get_assistant_vs_async(
        self, assistant_id: str
    ) -> List[ValidationInterface.VectorStoreRead]:
        resp = await self._request(
            "GET", f"/v1/assistants/{assistant_id}/vector-stores"
        )
        return [
            ValidationInterface.VectorStoreRead.model_validate(item) for item in resp
        ]

    async def _attach_vs_async(self, vector_store_id: str, assistant_id: str) -> bool:
        await self._request(
            "POST",
            f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/attach",
        )
        return True

    async def _detach_vs_async(self, vector_store_id: str, assistant_id: str) -> bool:
        await self._request(
            "DELETE",
            f"/v1/assistants/{assistant_id}/vector-stores/{vector_store_id}/detach",
        )
        return True

    # Sync facade helpers ------------------------------------------------ #
    def _run_sync(self, coro):
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise VectorStoreClientError("Sync call inside running loop")
        except RuntimeError:
            pass
        return asyncio.run(coro)

    # ──────────────────────────────────────────────────────────────────
    #  Helpers (private)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _normalise_hits(raw_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure each hit dict contains a top‑level 'meta_data' key so that all
        downstream components (reranker, synthesizer, envelope builder) can
        rely on a stable schema.
        """
        normalised: List[Dict[str, Any]] = []
        for h in raw_hits:
            md = h.get("meta_data") or h.get("metadata") or {}
            normalised.append(
                {
                    "text": h["text"],
                    "score": h["score"],
                    "meta_data": md,
                    "vector_id": h.get("vector_id"),
                    "store_id": h.get("store_id"),
                }
            )
        return normalised

    # Public API ---------------------------------------------------------- #
    def create_vector_store(
        self,
        name: str,
        *,
        vector_size: int = 384,
        distance_metric: str = "Cosine",
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreRead:
        """Create a new store owned by *this* API key."""
        return self._run_sync(
            self._create_vs_async(name, vector_size, distance_metric, config)
        )

    def create_vector_store_for_user(
        self,
        owner_id: str,
        name: str,
        *,
        vector_size: int = 384,
        distance_metric: str = "Cosine",
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreRead:
        """
        **Admin‑only** helper → create a store on behalf of *owner_id*.

        The caller’s API‑key must belong to an admin; otherwise the
        request will be rejected by the server with HTTP 403.
        """
        return self._run_sync(
            self._create_vs_for_user_async(
                owner_id, name, vector_size, distance_metric, config
            )
        )

    # ───────────────────────────────────────────────────────────────
    #  Convenience: ensure a per-user “file_search” store exists
    # ───────────────────────────────────────────────────────────────
    # unchanged … (get_or_create_file_search_store)

    def list_my_vector_stores(self) -> List[ValidationInterface.VectorStoreRead]:
        """List all non-deleted stores owned by *this* API-key’s user."""
        return self._run_sync(self._list_my_vs_async())

    # ───────────────────────────────────────────────────────────────
    #  NEW: real per-user listing (admin-only)
    # ───────────────────────────────────────────────────────────────
    async def _list_vs_by_user_async(self, user_id: str):
        resp = await self._request(
            "GET",
            "/v1/vector-stores/admin/by-user",
            params={"owner_id": user_id},
        )
        return [ValidationInterface.VectorStoreRead.model_validate(r) for r in resp]

    def get_stores_by_user(
        self,
        _user_id: str,
    ) -> List[ValidationInterface.VectorStoreRead]:  # noqa: ARG002
        """
        ⚠️ **Deprecated** – prefer impersonating the user’s API-key or using
        the newer RBAC endpoints, but keep working for legacy code.
        """
        warnings.warn(
            "`get_stores_by_user()` is deprecated; use `list_my_vector_stores()` or "
            "`VectorStoreClient(list_my_vector_stores)` with an impersonated key.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._run_sync(self._list_vs_by_user_async(_user_id))

    # ───────────────────────────────────────────────────────────────
    #  Convenience: ensure a per-user “file_search” store exists
    # ───────────────────────────────────────────────────────────────
    def get_or_create_file_search_store(self, user_id: Optional[str] = None) -> str:
        """
        Return the *oldest* vector-store named **file_search** for ``user_id``;
        create one if none exist.

        Parameters
        ----------
        user_id : Optional[str]
            • If **None**  → operate on *this* API-key’s stores
            • If not None → *admin-only*  – look up / create on behalf of ``user_id``

        Returns
        -------
        str
            The vector-store **id**.
        """

        # 1️⃣  Fetch candidate stores
        if user_id is None:
            # Normal user context – only see caller-owned stores
            stores = self.list_my_vector_stores()
        else:
            # Admin context – may inspect another user’s stores
            stores = self.get_stores_by_user(_user_id=user_id)

        file_search_stores = [s for s in stores if s.name == "file_search"]

        if file_search_stores:
            # 2️⃣  Pick the *earliest* (oldest created_at) to keep things stable
            chosen = min(
                file_search_stores,
                key=lambda s: (s.created_at or 0),
            )
            log.info(
                "Re-using existing 'file_search' store %s for user %s",
                chosen.id,
                user_id or "<self>",
            )
            return chosen.id

        # 3️⃣  Nothing found → create a fresh store
        if user_id is None:
            new_store = self.create_vector_store(name="file_search")
        else:
            # Requires admin API-key
            new_store = self.create_vector_store_for_user(
                owner_id=user_id,
                name="file_search",
            )

        log.info(
            "Created new 'file_search' store %s for user %s",
            new_store.id,
            user_id or "<self>",
        )
        return new_store.id

    def add_file_to_vector_store(
        self,
        vector_store_id: str,
        file_path: Union[str, Path],
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationInterface.VectorStoreFileRead:
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        return self._run_sync(self._add_file_async(vector_store_id, p, user_metadata))

    def delete_vector_store(
        self,
        vector_store_id: str,
        permanent: bool = False,
    ) -> Dict[str, Any]:
        return self._run_sync(self._delete_vs_async(vector_store_id, permanent))

    def delete_file_from_vector_store(
        self,
        vector_store_id: str,
        file_path: str,
    ) -> Dict[str, Any]:
        return self._run_sync(self._delete_file_async(vector_store_id, file_path))

    def list_store_files(
        self,
        vector_store_id: str,
    ) -> List[ValidationInterface.VectorStoreFileRead]:
        return self._run_sync(self._list_store_files_async(vector_store_id))

    def update_vector_store_file_status(
        self,
        vector_store_id: str,
        file_id: str,
        status: ValidationInterface.StatusEnum,
        error_message: Optional[str] = None,
    ) -> ValidationInterface.VectorStoreFileRead:
        return self._run_sync(
            self._update_file_status_async(
                vector_store_id, file_id, status, error_message
            )
        )

    def get_vector_stores_for_assistant(
        self,
        assistant_id: str,
    ) -> List[ValidationInterface.VectorStoreRead]:
        return self._run_sync(self._get_assistant_vs_async(assistant_id))

    def attach_vector_store_to_assistant(
        self,
        vector_store_id: str,
        assistant_id: str,
    ) -> bool:
        return self._run_sync(self._attach_vs_async(vector_store_id, assistant_id))

    def detach_vector_store_from_assistant(
        self,
        vector_store_id: str,
        assistant_id: str,
    ) -> bool:
        return self._run_sync(self._detach_vs_async(vector_store_id, assistant_id))

    def retrieve_vector_store_sync(
        self,
        vector_store_id: str,
    ) -> ValidationInterface.VectorStoreRead:
        resp = self._sync_api_client.get(f"/v1/vector-stores/{vector_store_id}")
        resp.raise_for_status()
        return ValidationInterface.VectorStoreRead.model_validate(resp.json())

    def vector_file_search_raw(
        self,
        vector_store_id: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        vector_store_host: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self._run_sync(
            self._search_vs_async(
                vector_store_id, query_text, top_k, filters, vector_store_host
            )
        )

    # ─────────────────────────────────────────────────────────────────────────────
    #  MID‑LEVEL: envelope but **no** rerank / synthesis
    # ─────────────────────────────────────────────────────────────────────────────
    def simple_vector_file_search(
        self,
        vector_store_id: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run a semantic search against *vector_store_id* and return the results
        wrapped in an OpenAI‑compatible envelope (file_search_call + assistant
        message with file_citation annotations).

        Args:
            vector_store_id: The store ID to query.
            query_text:      Natural‑language search text.
            top_k:           Maximum hits to retrieve.
            filters:         Optional Qdrant payload filter dict.

        Returns:
            dict: JSON‑serialisable envelope identical to the OpenAI format.
        """
        # 1️⃣  Raw hits (list[dict] from VectorStoreManager.query_store)
        raw_hits = self.vector_file_search_raw(
            vector_store_id=vector_store_id,
            query_text=query_text,
            top_k=top_k,
            filters=filters,
        )

        # 2️⃣  Normalise / enrich each hit so downstream code never crashes
        hits: List[Dict[str, Any]] = []
        for h in raw_hits:
            md = h.get("meta_data") or h.get("metadata") or {}
            hits.append(
                {
                    "text": h["text"],
                    "score": h["score"],
                    "meta_data": md,
                    "vector_id": h.get("vector_id"),
                    "store_id": h.get("store_id"),
                }
            )

        # 3️⃣  Generate human‑friendly answer text (LLM call or simple template)
        answer_text = summarize_hits(query_text, hits)

        # 4️⃣  Wrap everything into an OpenAI envelope
        return make_envelope(query_text, hits, answer_text)

    # ────────────────────────────────────────────────────────────────
    #  End‑to‑end: retrieve → (rerank) → synthesize → envelope
    # ────────────────────────────────────────────────────────────────
    def attended_file_search(
        self,
        vector_store_id: str,
        query_text: str,
        k: int = 20,
        vector_store_host: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a full file search with optional cross-encoder rerank and envelope synthesis.

        Parameters
        ----------
        vector_store_id : str
            The ID of the target vector store to query.
        query_text : str
            The natural-language search text.
        k : int, optional
            The maximum number of hits to retrieve (default is 20).
        vector_store_host : Optional[str], optional
            An optional override for the default vector store host.

        Returns
        -------
        Dict[str, Any]
            An OpenAI-style envelope containing the synthesized response.
        """

        # 1️⃣ Retrieve initial candidates (now with optional vector_store_host passthrough)
        hits = retriever.retrieve(
            self,
            vector_store_id=vector_store_id,
            query=query_text,
            k=k,
            vector_store_host=vector_store_host,
        )

        # 2️⃣ Optional cross-encoder / LLM rerank
        hits = reranker.rerank(query_text, hits, top_k=min(len(hits), 10))

        # 3️⃣ Normalize schema (guarantee 'meta_data')
        hits = self._normalise_hits(hits)

        # 4️⃣ Abstractive synthesis → OpenAI-style envelope
        return synthesize_envelope(
            query_text,
            hits,
            api_key=self.api_key,  # Project-David key
            base_url=self.base_url,  # Same backend
            provider_api_key=os.getenv("HYPERBOLIC_API_KEY"),  # Hyperbolic key
        )

    # ────────────────────────────────────────────────────────────────
    #  End‑to‑end: retrieve → (rerank) → synthesize → envelope
    # ────────────────────────────────────────────────────────────────
    def unattended_file_search(
        self,
        vector_store_id: str,
        query_text: str,
        k: int = 20,
        vector_store_host: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform a search over the file vector store and return normalized retrieval hits.

        This method executes a bare search pipeline: it retrieves vector-based candidates
        using semantic similarity, optionally applies reranking (e.g., cross-encoder or LLM-based),
        and normalizes the result schema. It does not perform synthesis or construct an OpenAI-style envelope.

        Use this when you want direct access to retrieved content for custom downstream handling,
        logging, inspection, or separate orchestration logic.

        Parameters
        ----------
        vector_store_id : str
            The ID of the vector store to search within.
        query_text : str
            The user query in natural language.
        k : int, optional
            The number of top hits to retrieve (default is 20).
        vector_store_host : Optional[str], optional
            Optional override for the vector store host (e.g., when calling remote Qdrant).

        Returns
        -------
        Dict[str, Any]
            A normalized list of retrieval results (each with metadata and score),
            without abstraction, synthesis, or formatting.
        """

        # 1️⃣ Retrieve initial candidates (now with optional vector_store_host passthrough)
        hits = retriever.retrieve(
            self,
            vector_store_id=vector_store_id,
            query=query_text,
            k=k,
            vector_store_host=vector_store_host,
        )

        # 2️⃣ Optional cross-encoder / LLM rerank
        hits = reranker.rerank(query_text, hits, top_k=min(len(hits), 10))

        # 3️⃣ Normalize schema (guarantee 'meta_data')
        hits = self._normalise_hits(hits)

        return hits
