# projectdavid — Python SDK

[![PyPI](https://img.shields.io/pypi/v/projectdavid)](https://pypi.org/project/projectdavid/)
[![Downloads](https://static.pepy.tech/badge/projectdavid)](https://pepy.tech/project/projectdavid)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Lint, Test, Tag, Publish Status](https://github.com/frankie336/projectdavid/actions/workflows/test_tag_release.yml/badge.svg)](https://github.com/frankie336/projectdavid/actions/workflows/test_tag_release.yml)

---




**The Python SDK for Project David — the open source, GDPR compliant successor to the OpenAI Assistants API.**

Same primitives. Every model. Your infrastructure.







---

## What is Project David?

Project David is a full-scale, containerized LLM orchestration platform built around the same primitives as the OpenAI Assistants API — **Assistants, Threads, Messages, Runs, and Tools** — but without the lock-in.

- **Provider agnostic** — Hyperbolic, TogetherAI, Ollama, or any OpenAI-compatible endpoint. Point at any inference provider and the platform normalizes the stream.
- **Every model** — hosted APIs today, raw local weights tomorrow. Bring your own model.
- **Your infrastructure** — fully self-hostable, open source, GDPR compliant, security audited.
- **Production grade** — sandboxed code execution (FireJail), multi-agent delegation, file serving with signed URLs, real-time streaming frontend.

> **Project Uni5** — the next milestone. `transformers`, GGUF, and vLLM adapters that mean a model straight off a training run has a full orchestration platform in minutes. From the lab to enterprise grade orchestration — instantly.

---


### Project Activity & Reach

| Metric | Status |
| :--- | :--- |
| **Total Downloads** | ![Total Downloads](https://static.pepy.tech/badge/projectdavid) |
| **Monthly Reach** | ![Monthly](https://img.shields.io/pypi/dm/projectdavid?color=blue&label=pypi%20downloads) |
| **Open Source Activity** | ![GitHub last commit](https://img.shields.io/github/last-commit/frankie336/projectdavid) |
| **Analytics** | [View Live Download Trends on ClickPy →](https://clickpy.clickhouse.com/dashboard/projectdavid) |

---


## Installation

```bash
pip install projectdavid
```

**Requirements:** Python 3.10+ · A running Project David platform instance

---

## Quick Start

```python
import os
from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

client = Entity(
    base_url=os.getenv("BASE_URL"),        # default: http://localhost:80
    api_key=os.getenv("ENTITIES_API_KEY"),
)

# Create an assistant
assistant = client.assistants.create_assistant(
    name="my_assistant",
    instructions="You are a helpful AI assistant.",
)

# Create a thread and send a message
thread = client.threads.create_thread()

message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content="Tell me about the latest trends in AI.",
    assistant_id=assistant.id,
)

# Create a run
run = client.runs.create_run(
    assistant_id=assistant.id,
    thread_id=thread.id,
)

# Stream the response
stream = client.synchronous_inference_stream
stream.setup(
    user_id=os.getenv("ENTITIES_USER_ID"),
    thread_id=thread.id,
    assistant_id=assistant.id,
    message_id=message.id,
    run_id=run.id,
    api_key=os.getenv("PROVIDER_API_KEY"),
)

for chunk in stream.stream_chunks(
    model="hyperbolic/deepseek-ai/DeepSeek-V3-0324",
    timeout_per_chunk=15.0,
):
    content = chunk.get("content", "")
    if content:
        print(content, end="", flush=True)
```

See the [Quick Start guide](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/sdk-quick-start.md) for the event-driven interface, tool calling, and advanced usage.

---

## Why Project David?

| | OpenAI Assistants API | LangChain | Project David |
|---|---|---|---|
| Assistants / Threads / Runs primitives | ✅ | ❌ | ✅ |
| Provider agnostic | ❌ | Partial | ✅ |
| Local model support | ❌ | Partial | ✅ |
| Raw weights → orchestration | ❌ | ❌ | ✅ *(Uni5)* |
| Sandboxed code execution | ✅ Black box | ❌ | ✅ FireJail PTY |
| Multi-agent delegation | Limited | ❌ | ✅ |
| Self-hostable | ❌ | ✅ | ✅ |
| GDPR compliant | ❌ | N/A | ✅ |
| Security audited | N/A | N/A | ✅ |
| Open source | ❌ | ✅ | ✅ |
| **Community Adoption** | Proprietary | High | ![Total Downloads](https://static.pepy.tech/badge/projectdavid) |
---

## Supported Inference Providers

[Full list of supported providers and endpoints →](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/providers/providers.md)

Works with any OpenAI-compatible endpoint out of the box — including Ollama for fully local inference.

---

## Environment Variables

| Variable | Description                                        |
|---|----------------------------------------------------|
| `ENTITIES_API_KEY` | Your Entities API key                              |
| `ENTITIES_USER_ID` | Your user ID                                       |
| `BASE_URL` | Platform base URL (default: `http://localhost:80`) |
| `PROVIDER_API_KEY` | Your inference provider API key                    |

---

## Documentation

| Topic | Link |
|---|---|
| Quick Start | [sdk-quick-start.md](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/sdk-quick-start.md) |
| Assistants | [sdk-assistants.md](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/sdk-assistants.md) |
| Threads | [sdk-threads.md](https://github.com/project-david-ai/docs/blob/main/src/pages/sdk/sdk-threads.md) |
| Messages | [sdk-messages.md](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/sdk-messages.md) |
| Runs | [sdk-runs.md](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/sdk-runs.md) |
| Inference | [sdk-inference.md](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/sdk-inference.md) |
| Tools | [sdk-tools.md](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/sdk-tools.md) |
| Function Calls | [function-calling-and-tool-execution.md](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/sdk-function-calls.md) |
| Code Interpreter | [sdk-code-interpreter.md](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/platform-tools/sdk-code-interpreter.md) |
| Files | [sdk-files.md](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/sdk-files.md) |
| Vector Store | [sdk-vector-store.md](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/sdk-vector-store.md) |

[Full SDK documentation →](https://github.com/project-david-ai/projectdavid_docs/tree/master/src/pages/sdk)

> Full hosted docs coming at `docs.projectdavid.co.uk`

---

## Related Repositories

| Repo | Description |
|---|---|
| [platform](https://github.com/project-david-ai/platform) | Core orchestration engine |
| [entities-common](https://github.com/project-david-ai/entities-common) | Shared utilities and validation |
| [david-core](https://github.com/project-david-ai/david-core) | Docker orchestration layer |
| [reference-frontend](https://github.com/project-david-ai/reference-frontend) | Reference streaming frontend |
| [entities_cook_book](https://github.com/project-david-ai/entities_cook_book) | Minimal tested examples — streaming, tools, search, stateful logic |

---


---

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)
