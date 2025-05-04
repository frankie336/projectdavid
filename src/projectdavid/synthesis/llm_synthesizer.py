# synthesis/llm_synthesizer.py
from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Any, Dict, List

from dotenv import load_dotenv

from ..utils.vector_search_formatter import make_envelope
from .prompt import SYSTEM_PROMPT, build_user_prompt

# ------------------------------------------------------------------ #
#  Environment & constants                                           #
# ------------------------------------------------------------------ #
load_dotenv()

USER_ID = os.getenv("ENTITIES_USER_ID")
MODEL = "hyperbolic/deepseek-ai/DeepSeek-V3-0324"
PROVIDER = "Hyperbolic"
MAX_TOKENS = 4096  # rough byte‑proxy budget

if TYPE_CHECKING:  # keeps IDE / MyPy happy
    from projectdavid import Entity  # noqa: F401

_ENTITIES_CLIENT = None  # will be initialised lazily


# ------------------------------------------------------------------ #
#  Helpers                                                          #
# ------------------------------------------------------------------ #
def _count_tokens(text: str) -> int:
    """Rough 4‑byte‑per‑token proxy. Replace with tokenizer if needed."""
    return len(text.encode("utf‑8")) // 4


# ------------------------------------------------------------------ #
#  Public API                                                       #
# ------------------------------------------------------------------ #
def synthesize_envelope(
    query: str,
    hits: List[Dict[str, Any]],
    top_n_ctx: int = 10,
) -> Dict[str, Any]:
    """
    Generate an abstractive answer + citations using your Hyperbolic‑backed
    model and return the OpenAI‑style envelope.
    """
    global _ENTITIES_CLIENT
    if _ENTITIES_CLIENT is None:  # ← lazy import breaks the cycle
        from projectdavid import Entity  # local import

        _ENTITIES_CLIENT = Entity(
            base_url=os.getenv("BASE_URL", "http://localhost:9000"),
            api_key=os.getenv("ENTITIES_API_KEY"),
        )

    # 1️⃣  Cull passages to fit budget
    ctx_passages, used = [], 0
    for h in hits[:top_n_ctx]:
        t = _count_tokens(h["text"])
        if used + t > MAX_TOKENS - 2048:  # reserve answer space
            break
        ctx_passages.append(h)
        used += t

    # 2️⃣  Build prompt
    user_prompt = build_user_prompt(query, ctx_passages)

    # 3️⃣  One‑shot assistant run via synchronous stream
    thread = _ENTITIES_CLIENT.threads.create_thread(participant_ids=[USER_ID])
    assistant = _ENTITIES_CLIENT.assistants.create_assistant(
        name="synth‑ephemeral",
        instructions=SYSTEM_PROMPT,
    )
    message = _ENTITIES_CLIENT.messages.create_message(
        thread_id=thread.id,
        role="user",
        content=user_prompt,
        assistant_id=assistant.id,
    )
    run = _ENTITIES_CLIENT.runs.create_run(
        assistant_id=assistant.id,
        thread_id=thread.id,
    )

    stream = _ENTITIES_CLIENT.synchronous_inference_stream
    stream.setup(
        user_id=USER_ID,
        thread_id=thread.id,
        assistant_id=assistant.id,
        message_id=message.id,
        run_id=run.id,
        api_key=os.getenv("HYPERBOLIC_API_KEY"),
    )

    # 4️⃣  Collect streamed chunks
    out = io.StringIO()
    for chunk in stream.stream_chunks(
        provider=PROVIDER,
        model=MODEL,
        timeout_per_chunk=60.0,
    ):
        out.write(chunk.get("content", ""))

    answer_text = out.getvalue().strip()

    # 5️⃣  Wrap in OpenAI‑style envelope
    return make_envelope(query, ctx_passages, answer_text)
