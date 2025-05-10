from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Dict, List, Optional

# third‑party
from dotenv import load_dotenv

# internal
from ..utils.vector_search_formatter import make_envelope
from .prompt import SYSTEM_PROMPT, build_user_prompt  # relative import

load_dotenv()

# ── defaults ─────────────────────────────────────────────────────────
DEFAULT_ASSISTANT = "plt_ast_mFySSaT11K0qM6RmFoOpW6"
DEFAULT_USER_ID = "user_hMcVBDyO810lkLw59RXvAS"
DEFAULT_MODEL = os.getenv("HYPERBOLIC_MODEL", "hyperbolic/deepseek-ai/DeepSeek-V3-0324")
DEFAULT_PROVIDER = os.getenv("HYPERBOLIC_PROVIDER", "Hyperbolic")
MAX_TOKENS = 4096

if TYPE_CHECKING:  # keep IDE / MyPy happy, avoid real import cycle
    from projectdavid import Entity  # noqa: F401

_ENTITIES_CLIENT: Optional["Entity"] = None  # lazy‑initialised singleton


# ── helper -----------------------------------------------------------
def _count_tokens(text: str) -> int:
    """Rough byte→token conversion (4bytes ≈ 1token)."""
    return len(text.encode()) // 4


# ── public API -------------------------------------------------------
def synthesize_envelope(
    query: str,
    hits: List[Dict[str, any]],
    *,
    api_key: str | None = None,  # Project‑David key
    base_url: str | None = None,
    provider_api_key: str | None = None,  # Hyperbolic key
    top_n_ctx: int = 10,
    user_id: str = DEFAULT_USER_ID,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, any]:
    # 1️⃣  Trim context to fit token budget
    ctx, used = [], 0
    for h in hits[:top_n_ctx]:
        t = _count_tokens(h["text"])
        if used + t > MAX_TOKENS - 2048:  # leave room for answer
            break
        ctx.append(h)
        used += t

    prompt = build_user_prompt(query, ctx)

    # 2️⃣  Lazy‑init Entities client (cycle‑safe)
    global _ENTITIES_CLIENT
    if _ENTITIES_CLIENT is None:
        from projectdavid import Entity  # local import

        _ENTITIES_CLIENT = Entity(
            base_url=base_url or os.getenv("BASE_URL", "http://localhost:9000"),
            api_key=api_key or os.getenv("ENTITIES_API_KEY"),
        )

    # 3️⃣  Spin up thread / assistant / run
    thread = _ENTITIES_CLIENT.threads.create_thread()

    msg = _ENTITIES_CLIENT.messages.create_message(
        thread_id=thread.id,
        role="user",
        content=prompt,
        assistant_id=DEFAULT_ASSISTANT,
    )

    run = _ENTITIES_CLIENT.runs.create_run(
        assistant_id=DEFAULT_ASSISTANT,
        thread_id=thread.id,
    )

    # 4️⃣  Stream the LLM response
    stream = _ENTITIES_CLIENT.synchronous_inference_stream
    stream.setup(
        user_id=user_id,
        thread_id=thread.id,
        assistant_id=DEFAULT_ASSISTANT,
        message_id=msg.id,
        run_id=run.id,
        api_key=provider_api_key or os.getenv("HYPERBOLIC_API_KEY"),
    )

    out = io.StringIO()
    for chunk in stream.stream_chunks(
        provider=provider,
        model=model,
        timeout_per_chunk=60.0,
    ):
        out.write(chunk.get("content", ""))

    answer_text = out.getvalue().strip()

    # 5️⃣  Replace raw file_id tokens with human‑friendly file_name
    for h in ctx:
        fid = h["meta_data"].get("file_id")
        fname = h["meta_data"].get("file_name")
        if fid and fname:
            # word‑boundary safe replacement
            answer_text = answer_text.replace(fid, fname)

    # 6️⃣  Clean up the thread to free resources
    try:
        _ENTITIES_CLIENT.threads.delete_thread(thread.id)
    except Exception as e:
        print(f"Failed to delete thread {thread.id}: {e}")

    # 7️⃣  Wrap into OpenAI‑style envelope with citations intact
    return make_envelope(query, ctx, answer_text)
