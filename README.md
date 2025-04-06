# Entities SDK
[![Test, Tag, Publish Status](https://github.com/frankie336/entitites_sdk/actions/workflows/test_tag_release.yml/badge.svg)](https://github.com/frankie336/entitites_sdk/actions/workflows/test_tag_release.yml)

The **Entities SDK** is a composable, Pythonic interface...



The **Entities SDK** is a composable, Pythonic interface to the [Entities API](https://github.com/frankie336/entities_api) for building intelligent applications across **local, open-source**, and **cloud LLMs**.

It unifies:

- Users, threads, assistants, messages, runs, inference
- **Function calling**, **code interpretation**, and **structured streaming**
- Vector memory, file uploads, and secure tool orchestration

Local inference is fully supported via [Ollama](https://github.com/ollama).

---

## 🔌 Supported Inference Providers

| Provider                                         | Type                     |
|--------------------------------------------------|--------------------------|
| [Ollama](https://github.com/ollama)              |  **Local** (Self-Hosted) |
| [DeepSeek](https://platform.deepseek.com/)       | ☁ **Cloud** (Open-Source) |
| [Hyperbolic](https://hyperbolic.xyz/)            | ☁ **Cloud** (Proprietary) |
| [OpenAI](https://platform.openai.com/)           | ☁ **Cloud** (Proprietary) |
| [Together AI](https://www.together.ai/)          | ☁ **Cloud** (Aggregated) |
| [Azure Foundry](https://azure.microsoft.com)     | ☁ **Cloud** (Enterprise) |

---

## 📦 Installation

```bash
pip install entities

```

---

##  Quick Start

```python
from entities import Entities
import os
from dotenv import load_dotenv

load_dotenv()

client = Entities(
    base_url='http://localhost:9000',
    api_key=os.getenv("API_KEY")
)

user = client.users.create_user(name="demo_user")
thread = client.threads.create_thread(participant_ids=[user.id])
assistant = client.assistants.create_assistant(name="Demo Assistant")

message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content="Hello, assistant!",
    assistant_id=assistant.id
)

run = client.runs.create_run(
    assistant_id=assistant.id,
    thread_id=thread.id
)

stream = client.inference.stream_inference_response(
    provider="Hyperbolic",
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    thread_id=thread.id,
    message_id=message.id,
    run_id=run.id,
    assistant_id=assistant.id
)

for chunk in stream:
    print(chunk)
```

---

## 📚 Documentation

| Domain              | Link                                                   |
|---------------------|--------------------------------------------------------|
| Assistants          | [assistants.md](/docs/assistants.md)                   |
| Threads             | [threads.md](/docs/threads.md)                         |
| Messages            | [messages.md](/docs/messages.md)                       |
| Runs                | [runs.md](/docs/runs.md)                               |
| Inference           | [inference.md](/docs/inference.md)                     |
| Streaming           | [streams.md](/docs/streams.md)                         |
| Function Calling    | [function_calling.md](/docs/function_calling.md)       |
| Code Interpretation | [code_interpretation.md](/docs/code_interpretation.md) |
| Files               | [files.md](/docs/files.md)                             |
| Vector Store        | [vector_store.md](/docs/vector_store.md)               |
| Versioning          | [versioning.md](/docs/versioning.md)                   |

---

## ✅ Compatibility & Requirements

- Python **3.8+**
- Compatible with **local** or **cloud** deployments of the Entities API

---

## 🌍 Related Repositories

- 🔌 [Entities API](https://github.com/frankie336/entities_api) — containerized API backend
- 
- 📚 [entities_common](https://github.com/frankie336/entities_common) — shared validation, schemas, utilities, and tools.
      This package is auto installed as dependency of entities SDK or entities API.
