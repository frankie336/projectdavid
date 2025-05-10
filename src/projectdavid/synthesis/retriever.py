"""
Light‑weight retrieval helper.

* Accepts **any** client that exposes `vector_file_search_raw(...)`
* Returns the raw ANN hits (each already containing meta_data)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

# ── Conditional import only for type‑checkers / IDEs ───────────────────────
if TYPE_CHECKING:  # pragma: no cover
    from ..clients.vectors import VectorStoreClient


# ── Structural type so MyPy / static analysis knows the shape ─────────────
class VectorStoreLike(Protocol):
    def vector_file_search_raw(
        self,
        vector_store_id: str,
        query_text: str,
        top_k: int = 5,
        filters: Dict | None = None,
        vector_store_host: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...


# ── Public helper ─────────────────────────────────────────────────────────
def retrieve(
    client: "VectorStoreLike",
    vector_store_id: str,
    query: str,
    k: int = 20,
    filters: Optional[Dict] = None,
    vector_store_host: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run a raw similarity search against *vector_store_id*.

    The underlying client must implement **vector_file_search_raw**; results
    already include any chunk‑level metadata (page, lines, file_id, …).

    Parameters
    ----------
    client:
        Any object that satisfies the ``VectorStoreLike`` protocol.
    vector_store_id:
        The target vector store to query.
    query:
        Natural‑language search string.
    k:
        Number of top passages to return (default 20).
    filters:
        Optional Qdrant payload‑filter dictionary.
    vector_store_host:
        Optionally override the default vector store host for this query.

    Returns
    -------
    List[Dict[str, Any]]
        Raw hit dictionaries in the standard Project‑David schema.
    """
    return client.vector_file_search_raw(
        vector_store_id=vector_store_id,
        query_text=query,
        top_k=k,
        filters=filters,
        vector_store_host=vector_store_host,
    )
