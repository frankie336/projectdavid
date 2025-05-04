from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

_cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


def rerank(query: str, hits: List[Dict[str, Any]], top_k: int = 10) -> List[Dict]:
    pairs = [[query, h["text"]] for h in hits]
    scores = _cross.predict(pairs, convert_to_tensor=False)
    for h, sc in zip(hits, scores):
        h["re_score"] = float(sc)
    return sorted(hits, key=lambda x: x["re_score"], reverse=True)[:top_k]
