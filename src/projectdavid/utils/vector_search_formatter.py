from typing import Any, Dict, List

from projectdavid_common import UtilsInterface
from projectdavid_common.validation import (
    AssistantMessage,
    FileCitation,
    FileSearchCall,
    FileSearchEnvelope,
    OutputText,
)

_id = UtilsInterface.IdentifierService()


def make_envelope(query: str, hits: List[Dict[str, Any]], answer_text: str) -> dict:
    """Wrap search results in OpenAI‑style envelope."""
    # Build citations (one per hit here; de‑dupe as you wish)
    citations: List[FileCitation] = []
    for hit in hits:
        file_id = hit["meta_data"]["file_id"]
        filename = hit["meta_data"]["file_name"]
        # naive locate — first occurrence of filename in answer
        offset = answer_text.find(filename)
        citations.append(FileCitation(index=offset, file_id=file_id, filename=filename))

    fs_call = FileSearchCall(
        id=_id.generate_prefixed_id("fs"),  # e.g. fs_<uuid>
        queries=[query],
    )

    assistant_msg = AssistantMessage(
        id=_id.generate_prefixed_id("msg"),
        content=[OutputText(text=answer_text, annotations=citations)],
    )

    return FileSearchEnvelope(output=[fs_call, assistant_msg]).model_dump()
