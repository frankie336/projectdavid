from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


def rerank(query: str, hits: List[Dict[str, Any]], top_k: int = 10) -> List[Dict]:
    try:
        cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
            local_files_only=True,  # ensures no unexpected downloads server-side
        )
    except Exception as e:
        import warnings

        warnings.warn(
            f"CrossEncoder model unavailable: {e}. Returning original hits.",
            RuntimeWarning,
        )
        return hits[:top_k]

    pairs = [[query, h["text"]] for h in hits]
    scores = cross_encoder.predict(pairs, convert_to_tensor=False)

    for h, sc in zip(hits, scores):
        h["re_score"] = float(sc)

    return sorted(hits, key=lambda x: x["re_score"], reverse=True)[:top_k]
