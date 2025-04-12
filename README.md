# Entity  — by Project David

[![Test, Tag, Publish Status](https://github.com/frankie336/entitites_sdk/actions/workflows/test_tag_release.yml/badge.svg)](https://github.com/frankie336/entitites_sdk/actions/workflows/test_tag_release.yml)

The **Entity SDK** is a composable, Pythonic interface to the [Entities API](https://github.com/frankie336/entities_api) for building intelligent applications across **local, open-source**, and **cloud LLMs**.

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
pip install projectdavid

```

---

##  Quick Start

**Create a user**

```python
from projectdavid import Entity
from dotenv import load_dotenv


# In dev environments the base url will default 
# to http://localhost:9000
# In prod encvironments, you need to set it to your FDQN

client = Entity(
      base_url='http://localhost:9000',
      api_key=os.getenv("API_KEY")
)
user = client.users.create_user(name='test_user')
```

```bash
print(user.id)
user_s1xkwzViWkq0dqUBGri9EU
```




```python

assistant = client.assistants.create_assistant(name='test_assistant',
                                               instructions='You are a helpful AI assistant',

                                               )

```

```bash
print(assistant.id)
asst_3SrhB8vCFTl56M0dtjbqyV
```

The above steps can be repeated at whatever scale and frequency needed: you can create unlimted users,
and unlimited asistants

---

**Setting up a prompt and handling response**


- Entities supports synchronous streams



```python

# step 1 - Create a thread  

thread = client.threads.create_thread(participant_ids=["user_s1xkwzViWkq0dqUBGri9EU"])


# step 2 - Create a message 

message = client.messages.create_message(
      thread_id="thread_kn7vwRPfutWyvwl4um1VCV",
      role="user",
      content="Hello, assistant!",
      assistant_id="asst_3SrhB8vCFTl56M0dtjbqyV"
)

# step 3 - Create a run 

run = client.runs.create_run(
      assistant_id="asst_3SrhB8vCFTl56M0dtjbqyV",
      thread_id="thread_kn7vwRPfutWyvwl4um1VCV"
)


# Instantiate the syncronous streaming helper 

sync_stream = client.synchronous_inference_stream


# step 4 - Set up the stream

sync_stream.setup(
            user_id="user_s1xkwzViWkq0dqUBGri9EU",
            thread_id="thread_kn7vwRPfutWyvwl4um1VCV",
            assistant_id="default",
            message_id=message.id,
            run_id=run.id,

        )

# step 5 - Stream the response

# Stream completions synchronously

# The api_key param is optional but needed if you are usign
# a cloud inference providider 

import logging
import json

logging.basicConfig(level=logging.INFO)

# Stream completions synchronously
logging.info("Beginning sync stream...")
for chunk in sync_stream.stream_chunks(
    provider="Hyperbolic",
    model="hyperbolic/deepseek-ai/DeepSeek-V3",
    timeout_per_chunk=15.0,
    api_key='your-hyperbolic-key-here',
):
    logging.info(json.dumps(chunk, indent=2))

logging.info("Stream finished.")

```


A snip  of the stream strcuture:

```bash

INFO:root:{
  "status": "handshake"
}
INFO:root:{
  "status": "initializing"
}
INFO:root:{
  "status": "processing"
}
INFO:root:{
  "type": "content",
  "content": "Hello",
  "first_chunk": true
}
INFO:root:{
  "type": "content",
  "content": "!"
}
INFO:root:{
  "type": "content",
  "content": " It"
}
INFO:root:{
  "type": "content",
  "content": "'s"
}
INFO:root:{
  "type": "content",
  "content": " great"
}
INFO:root:{
  "type": "content",
  "content": " to"
}
INFO:root:{
  "type": "content",
  "content": " hear"
}
...
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
| Function Calling    | [function_calls.md](/docs/function_calls.md)       |
| Code Interpretation | [code_interpretation.md](/docs/code_interpretation.md) |
| Files               | [files.md](/docs/files.md)                             |
| Vector Store(RAG)   | [vector_store.md](/docs/vector_store.md)               |
| Versioning          | [versioning.md](/docs/versioning.md)                   |

---

## ✅ Compatibility & Requirements

- Python **3.10+**
- Compatible with **local** or **cloud** deployments of the Entities API

---

## 🌍 Related Repositories

- 🔌 [Entities API](https://github.com/frankie336/entities_api) — containerized API backend
- 
- 📚 [entities_common](https://github.com/frankie336/entities_common) — shared validation, schemas, utilities, and tools.
      This package is auto installed as dependency of entities SDK or entities API.
