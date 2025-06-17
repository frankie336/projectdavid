from sympy import python

# Inference

## Overview

Inference is the final stage of the Entities API workflow, where the assistant processes a prompt and generates a response. This stage highlights the assistant's intelligence and capabilities. Inference can be executed on edge devices or in the cloud, according to your specific needs. Our API supports both options, allowing flexibility tailored to your use case.

---

## Basic Inference Streaming Example

The following example demonstrates how to:

1. Create an assistant and a thread.
2. Send a user message.
3. Initiate a run.
4. Stream the assistant's response via the Hyperbolic provider.

### Requirements

Ensure the following environment variables are set:

- `ENTITIES_API_KEY`: API key for Entities API access.
- `BASE_URL`: Base URL of the Entities API instance (default: `http://localhost:9000`).
- `HYPERBOLIC_API_KEY`: API key for the Hyperbolic provider.
- `ENTITIES_USER_ID`: User ID associated with the Entities API.

### Example Implementation

```python
import os
from dotenv import load_dotenv
from projectdavid import Entity

# Load environment variables from .env file
load_dotenv()

# Initialize Entities API client
client = Entity(
    base_url=os.getenv("BASE_URL", "http://localhost:9000"),
    api_key=os.getenv("ENTITIES_API_KEY")
)

# Constants for Hyperbolic provider
API_KEY = "your-hyperbolic-key-here"

MODEL = "hyperbolic/deepseek-ai/DeepSeek-V3-0324"
PROVIDER = "Hyperbolic"

def main():

    # Create assistant (can be reused across runs)
    assistant = client.assistants.create_assistant(
        name="test_assistant",
        instructions="You are a helpful AI assistant",)

    # Create thread (can also be reused)
    thread = client.threads.create_thread()


    # Create user message
    message = client.messages.create_message(
        thread_id=thread.id,
        role="user",
        content="Explain a black hole to me in pure mathematical terms",
        assistant_id=assistant.id
    )

    # Create a run
    run = client.runs.create_run(
        assistant_id=assistant.id,
        thread_id=thread.id
    )
```

### Stream the Assistant's Response

```python

    # --------------------------------------
    # 
    # Setup synchronous streaming
    #
    #-----------------------------------------
    sync_stream = client.synchronous_inference_stream
    sync_stream.setup(
        user_id=user_id,
        thread_id=thread.id,
        assistant_id=assistant.id,
        message_id=message.id,
        run_id=run.id,
        api_key=API_KEY
    )

    # Stream the assistant's response
    
    
    try:
        for chunk in sync_stream.stream_chunks(
            provider=PROVIDER,
            model=MODEL,
            timeout_per_chunk=60.0,
            suppress_fc=True,
        ):
            content = chunk.get("content", "")
            if content:
                print(content, end="", flush=True)
        print("\n--- End of Stream ---")
    except Exception as e:
        print(f"Stream Error: {e}")

if __name__ == "__main__":
    main()

```