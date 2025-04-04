# Entities SDK

The **Entities** SDK provides a Pythonic interface to the [Entities AI API](https://github.com/frankie336/entities_api).  
It offers a unified abstraction for building applications that interact with open-source and cloud-based LLMs, including local inference via [Ollama](https://github.com/ollama).

It supports advanced capabilities like multi-turn dialogue, [function calling](/docs/function_calling.md), [code interpretation](/docs/code_interpretation.md), and streaming—all through a consistent API surface.


---

## 🔌 Supported Inference Providers

| Provider                                         | Type                        |
|--------------------------------------------------|-----------------------------|
| [Ollama](https://github.com/ollama)              | **Local** (Self-Hosted)     |
| [DeepSeek](https://platform.deepseek.com/)       | **Cloud** (Open-Source)     |
| [Hyperbolic](https://hyperbolic.xyz/)            | **Cloud** (Proprietary)     |
| [OpenAI](https://platform.openai.com/)           | **Cloud** (Proprietary)     |
| [Together AI](https://www.together.ai/)          | **Cloud** (Aggregated)      |
| [MS Azure Foundry](https://azure.microsoft.com)  | **Cloud** (Enterprise)      |

---

## 🧠 Why Entities API?

The modern inference landscape is fragmented—each provider offers its own keys, schemas, endpoints, and semantics.

**Entities** abstracts this chaos into a unified, state-aware assistant framework.  
You gain consistency and flexibility across providers, with tools that help you orchestrate assistants, tools, messages, runs, threads, and memory.

---

## 🧾 Dialogue State Management

LLM applications require explicit management of multi-turn dialogue state.

Example:

```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What’s the capital of France?"},
  {"role": "assistant", "content": "The capital of France is Paris."},
  {"role": "user", "content": "What’s the population of Paris?"},
  {"role": "assistant", "content": "As of the latest data, the population of Paris is approximately 2.1 million."}
]
