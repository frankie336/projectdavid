# synthesis/llm_synthesizer.py
import io
import itertools
import json

# ------------------------------------------------------------------ #
#  Configure once (env vars come from .env as in your demo script)   #
# ------------------------------------------------------------------ #
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from projectdavid import Entity  # your SDK

from ..utils.vector_search_formatter import make_envelope
from .prompt import SYSTEM_PROMPT, build_user_prompt

load_dotenv()

_ENTITIES_CLIENT = Entity(
    base_url=os.getenv("BASE_URL", "http://localhost:9000"),
    api_key=os.getenv("ENTITIES_API_KEY"),
)

USER_ID = os.getenv("ENTITIES_USER_ID")  # for streams
MODEL = "hyperbolic/deepseek-ai/DeepSeek-V3-0324"
PROVIDER = "Hyperbolic"
MAX_TOKENS = 4096  # budget guard


def _count_tokens(text: str) -> int:
    # Cheap byte‑level proxy; adjust if you wire in a tokenizer later
    return len(text.encode("utf‑8")) // 4  # rough 4‑byte/ token


# ------------------------------------------------------------------ #
#  Public API: synthesize_envelope()                                 #
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
    # 1️⃣  Pick passages that fit the token/byte budget
    ctx_passages, used = [], 0
    for h in hits[:top_n_ctx]:
        t = _count_tokens(h["text"])
        if used + t > MAX_TOKENS - 2048:  # leave headroom for answer
            break
        ctx_passages.append(h)
        used += t

    # 2️⃣  Build the composite user prompt
    user_prompt = build_user_prompt(query, ctx_passages)

    # 3️⃣  Kick off a “one‑off assistant” run via synchronous stream
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

    # 4️⃣  Collect streamed chunks into the final answer text
    out = io.StringIO()
    for chunk in stream.stream_chunks(
        provider=PROVIDER,
        model=MODEL,
        timeout_per_chunk=60.0,
    ):
        out.write(chunk.get("content", ""))

    answer_text = out.getvalue().strip()

    # 5️⃣  Wrap in OpenAI‑style envelope with citations
    return make_envelope(query, ctx_passages, answer_text)
