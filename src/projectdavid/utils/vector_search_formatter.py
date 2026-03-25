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
    """
    Wrap a vector‑store search in an OpenAI‑style envelope.

    * De‑duplicates citations so each file appears once.
    * Adds optional page + line numbers when present in hit.meta_data.
    * Computes char‑offset for each filename inside the generated answer text.
    """
    citations: List[FileCitation] = []
    seen_files = set()

    for hit in hits:
        md = hit["meta_data"]
        file_id = md["file_id"]

        # Skip duplicates
        if file_id in seen_files:
            continue
        seen_files.add(file_id)

        filename = md["file_name"]

        # Locate first occurrence of the filename in answer_text (-1 if absent)
        offset = answer_text.find(filename)

        citations.append(
            FileCitation(
                index=offset,
                file_id=file_id,
                filename=filename,
                page=md.get("page"),  # present for PDF chunks
                lines=md.get("lines"),  # present for PDF chunks
            )
        )

    fs_call = FileSearchCall(
        id=_id.generate_prefixed_id("fs"),  # e.g. fs_<uuid>
        queries=[query],
    )

    assistant_msg = AssistantMessage(
        id=_id.generate_prefixed_id("msg"),
        content=[OutputText(text=answer_text, annotations=citations)],
    )

    return FileSearchEnvelope(output=[fs_call, assistant_msg]).model_dump()
