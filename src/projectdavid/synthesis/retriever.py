from typing import TYPE_CHECKING, Any, Dict, List, Protocol

# ── Conditional import only for static analysis / IDEs ──────────────────
if TYPE_CHECKING:
    from ..clients.vectors import VectorStoreClient


# ── Lightweight structural type so MyPy still knows the shape ───────────
class VectorStoreLike(Protocol):
    def search_vector_store(  # noqa: D401
        self,
        vector_store_id: str,
        query_text: str,
        top_k: int = 5,
        filters: Dict | None = None,
    ) -> List[Dict[str, Any]]: ...


# ── Public helper ───────────────────────────────────────────────────────
def retrieve(
    client: "VectorStoreLike",
    vector_store_id: str,
    query: str,
    k: int = 20,
    filters: Dict | None = None,
) -> List[Dict[str, Any]]:
    """
    Raw similarity search (already includes page & line in meta_data).

    Args:
        client:          Any object implementing `search_vector_store`.
        vector_store_id: The store to query.
        query:           Natural‑language question.
        k:               Number of passages to return.
        filters:         Optional Qdrant payload filter.

    Returns:
        List of hit dicts enriched with meta_data.
    """
    return client.search_vector_store(
        vector_store_id=vector_store_id,
        query_text=query,
        top_k=k,
        filters=filters,
    )
