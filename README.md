# projectdavid — Entity SDK

[![Lint, Test, Tag, Publish Status](https://github.com/frankie336/projectdavid/actions/workflows/test_tag_release.yml/badge.svg)](https://github.com/frankie336/projectdavid/actions/workflows/test_tag_release.yml)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

Python SDK for the Entities API — build, run, and coordinate AI assistants across local and cloud inference providers.

## Installation

```bash
pip install projectdavid
```

## Requirements

- Python 3.10+
- An Entities API key (`ENTITIES_API_KEY`)

## Supported Inference Providers


## Quick Start

```python
import os
from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

client = Entity(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("ENTITIES_API_KEY"),
)

assistant = client.assistants.create_assistant(
    name="my_assistant",
    instructions="You are a helpful AI assistant.",
)

thread = client.threads.create_thread()

message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content="Tell me about the latest trends in AI.",
    assistant_id=assistant.id,
)

run = client.runs.create_run(
    assistant_id=assistant.id,
    thread_id=thread.id,
)

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

See the [Quick Start guide](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/sdk/sdk-quick-start.md) for the event-driven interface and advanced usage.

## Environment Variables

| Variable | Description |
|---|---|
| `ENTITIES_API_KEY` | Your Entities API key |
| `ENTITIES_USER_ID` | Your user ID |
| `BASE_URL` | API base URL (default: `http://localhost:9000`) |
| `PROVIDER_API_KEY` | Your inference provider API key |

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

## Related Repositories

| Repo | Description |
|---|---|
| [platform](https://github.com/project-david-ai/platform) | Core orchestration engine |
| [entities-common](https://github.com/project-david-ai/entities-common) | Shared utilities and validation |
| [david-core](https://github.com/project-david-ai/david-core) | Docker orchestration layer |

> When the hosted docs site is live, all documentation links will be updated to `docs.projectdavid.co.uk`.