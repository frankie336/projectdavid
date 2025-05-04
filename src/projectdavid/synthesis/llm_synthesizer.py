from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Dict, List, Optional

# third‑party
from dotenv import load_dotenv

from ..utils.vector_search_formatter import make_envelope

# internal / relative
from .prompt import SYSTEM_PROMPT, build_user_prompt  # ←  relative “.prompt”

load_dotenv()

DEFAULT_USER_ID = "user_TUofmVbbjzFyMHo3YIihEv"
DEFAULT_MODEL = os.getenv("HYPERBOLIC_MODEL", "hyperbolic/deepseek-ai/DeepSeek-V3-0324")
DEFAULT_PROVIDER = os.getenv("HYPERBOLIC_PROVIDER", "Hyperbolic")
MAX_TOKENS = 4096

if TYPE_CHECKING:
    from projectdavid import Entity

_ENTITIES_CLIENT: Optional["Entity"] = None


# ---------- helpers ------------------------------------------------- #
def _count_tokens(text: str) -> int:
    return len(text.encode()) // 4


# ---------- main ---------------------------------------------------- #
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

    # 1. trim context
    ctx, used = [], 0
    for h in hits[:top_n_ctx]:
        t = _count_tokens(h["text"])
        if used + t > MAX_TOKENS - 2048:
            break
        ctx.append(h)
        used += t

    prompt = build_user_prompt(query, ctx)

    # 2. lazy‑init Entities client (cycle‑safe)
    global _ENTITIES_CLIENT
    if _ENTITIES_CLIENT is None:
        from projectdavid import Entity

        _ENTITIES_CLIENT = Entity(
            base_url=base_url or os.getenv("BASE_URL", "http://localhost:9000"),
            api_key=api_key or os.getenv("ENTITIES_API_KEY"),
        )

    # 3. thread / assistant / run
    thread = _ENTITIES_CLIENT.threads.create_thread(participant_ids=[user_id])
    assistant = _ENTITIES_CLIENT.assistants.create_assistant(
        name="synth‑ephemeral",
        instructions=SYSTEM_PROMPT,
    )
    msg = _ENTITIES_CLIENT.messages.create_message(
        thread_id=thread.id,
        role="user",
        content=prompt,
        assistant_id=assistant.id,
    )
    run = _ENTITIES_CLIENT.runs.create_run(
        assistant_id=assistant.id,
        thread_id=thread.id,
    )

    # 4. stream
    stream = _ENTITIES_CLIENT.synchronous_inference_stream
    stream.setup(
        user_id=user_id,
        thread_id=thread.id,
        assistant_id=assistant.id,
        message_id=msg.id,
        run_id=run.id,
        api_key=provider_api_key or os.getenv("HYPERBOLIC_API_KEY"),
    )

    out = io.StringIO()
    for chunk in stream.stream_chunks(
        provider=provider, model=model, timeout_per_chunk=60.0
    ):
        out.write(chunk.get("content", ""))

    return make_envelope(query, ctx, out.getvalue().strip())
