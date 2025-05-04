from typing import Any, Dict, List

SYSTEM_PROMPT = """You are an expert research assistant.
Answer the user question using ONLY the provided excerpts.
After each claim, cite the supporting file like (citation: <file_id>).
Do NOT fabricate information.
"""


# synthesis/prompt.py
def build_user_prompt(query: str, passages: List[Dict[str, Any]]) -> str:
    segs = []
    for i, p in enumerate(passages, 1):
        # tolerate either meta_data or metadata
        meta = p.get("meta_data") or p.get("metadata") or {}
        page = f" page {meta['page']}" if meta.get("page") else ""
        segs.append(
            f"[{i}] ({meta.get('file_id','?')}{page}) {p['text'][:500].strip()}"
        )
    joined = "\n\n".join(segs)
    return f"User question: {query}\n\nExcerpts:\n{joined}\n\nAnswer:"
