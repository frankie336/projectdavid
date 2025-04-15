# Entity  — by Project David

[![Lint, Test, Tag, Publish Status](https://github.com/frankie336/projectdavid/actions/workflows/test_tag_release.yml/badge.svg)](https://github.com/frankie336/entitites_sdk/actions/workflows/test_tag_release.yml)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

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

```python
import os

from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

# --------------------------------------------------
# Load the Entities client with your user API key
# Note: if you define ENTITIES_API_KEY="ea_6zZiZ..."
# in .env, you do not need to pass in the API key directly.
# We pass in here directly for clarity
# ---------------------------------------------------
client = Entity(base_url="http://localhost:9000", api_key=os.getenv("ENTITIES_API_KEY"))

user_id = "user_kUKV8octgG2aMc7kxAcD3i"

# -----------------------------
# create an assistant
# ------------------------------
assistant = client.assistants.create_assistant(
    name="test_assistant",
    instructions="You are a helpful AI assistant",
)
print(f"created assistant with ID: {assistant.id}")

# -----------------------------------------------
# Create a thread
# Note:
# - Threads are re-usable
# Reuse threads in the case you want as continued
# multi turn conversation
# ------------------------------------------------
print("Creating thread...")
thread = client.threads.create_thread(participant_ids=[user_id])

print(f"created thread with ID: {thread.id}")
# Store the dynamically created thread ID
actual_thread_id = thread.id


# -----------------------------------------
#  Create a message using the NEW thread ID
# --------------------------------------------
print(f"Creating message in thread {actual_thread_id}...")
message = client.messages.create_message(
    thread_id=actual_thread_id,
    role="user",
    content="Hello, assistant! Tell me about the latest trends in AI.",
    assistant_id=assistant.id,
)
print(f"Created message with ID: {message.id}")

# ---------------------------------------------
# step 3 - Create a run using the NEW thread ID
# ----------------------------------------------
print(f"Creating run in thread {actual_thread_id}...")
run = client.runs.create_run(assistant_id=assistant.id, thread_id=actual_thread_id)
print(f"Created run with ID: {run.id}")

# ------------------------------------------------
# Instantiate the synchronous streaming helper
# --------------------------------------------------
sync_stream = client.synchronous_inference_stream

# ------------------------------------------------------
# step 4 - Set up the stream using the NEW thread ID
# --------------------------------------------------------
print(f"Setting up stream for thread {actual_thread_id}...")
sync_stream.setup(
    user_id=user_id,
    thread_id=actual_thread_id,
    assistant_id=assistant.id,
    message_id=message.id,
    run_id=run.id,
    api_key=os.getenv("HYPERBOLIC_API_KEY"),
)
print("Stream setup complete. Starting streaming...")

# --- Stream initial LLM response ---
try:
    for chunk in sync_stream.stream_chunks(
        provider="Hyperbolic",
        model="hyperbolic/deepseek-ai/DeepSeek-V3-0324",  # Ensure this model is valid/available
        timeout_per_chunk=15.0,
    ):
        content = chunk.get("content", "")
        if content:
            print(content, end="", flush=True)
    print("\n--- End of Stream ---")  # Add newline after stream
except Exception as e:
    print(f"\n--- Stream Error: {e} ---")  # Catch errors during streaming

print("Script finished.")
```

**The assisants  response**:


Hello! The field of AI is evolving rapidly, and here are some of the latest trends as of early 2025:

### 1. **Multimodal AI Models**  
   - Models like GPT-4, Gemini, and others now seamlessly process text, images, audio, and video in a unified way, enabling richer interactions (e.g., ChatGPT with vision).  
   - Applications include real-time translation with context, AI-generated video synthesis, and more immersive virtual assistants.

### 2. **Smaller, More Efficient Models**  
   - While giant models (e.g., GPT-4, Claude 3) still dominate, there’s a push for smaller, specialized models (e.g., Microsoft’s Phi-3, Mistral 7B) that run locally on devices with near-LLM performance.  
   - Focus on **energy efficiency** and reduced computational costs.

### 3. **AI Agents & Autonomous Systems**  
   - AI “agents” (e.g., OpenAI’s “Agentic workflows”) can now perform multi-step tasks autonomously, like coding, research, or booking trips.  
   - Companies are integrating agentic AI into workflows (e.g., Salesforce, Notion AI).

### 4. **Generative AI Advancements**  
   - **Video generation**: Tools like OpenAI’s Sora, Runway ML, and Pika Labs produce high-quality, longer AI-generated videos.  
   - **3D asset creation**: AI can now generate 3D models from text prompts (e.g., Nvidia’s tools).  
   - **Voice cloning**: Ultra-realistic voice synthesis (e.g., ElevenLabs) is raising ethical debates.

### 5. **Regulation & Ethical AI**  
   - Governments are catching up with laws like the EU AI Act and U.S. executive orders on AI safety.  
   - Watermarking AI content (e.g., C2PA standards) is gaining traction to combat deepfakes.

### 6. **AI in Science & Healthcare**  
   - AlphaFold 3 (DeepMind) predicts protein interactions with unprecedented accuracy.  
   - AI-driven drug discovery (e.g., Insilico Medicine) is accelerating clinical trials.

### 7. **Open-Source vs. Closed AI**  
   - Tension between open-source (Mistral, Meta’s Llama 3) and proprietary models (GPT-4, Gemini) continues, with debates over safety and innovation.

### 8. **AI Hardware Innovations**  
   - New chips (e.g., Nvidia’s Blackwell, Groq’s LPUs) are optimizing speed and cost for AI workloads.  
   - “AI PCs” with NPUs (neural processing units) are becoming mainstream.

### 9. **Personalized AI**  
   - Tailored AI assistants learn individual preferences (e.g., Rabbit R1, Humane AI Pin).  
   - Privacy-focused local AI (e.g., Apple’s on-device AI in iOS 18).

### 10. **Quantum AI (Early Stages)**  
   - Companies like Google and IBM are exploring quantum machine learning, though practical applications remain limited.

Would you like a deeper dive into any of these trends?

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
