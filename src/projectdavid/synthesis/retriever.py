from typing import Any, Dict, List

from ..clients.vectors import VectorStoreClient


def retrieve(
    client: VectorStoreClient,
    vector_store_id: str,
    query: str,
    k: int = 20,
    filters=None,
) -> List[Dict[str, Any]]:
    """Raw similarity search (already includes page & line in meta_data)."""
    return client.search_vector_store(
        vector_store_id=vector_store_id,
        query_text=query,
        top_k=k,
        filters=filters,
    )
