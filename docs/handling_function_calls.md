# Function Calling and Tool Execution (Unified Loop)

## Overview

Most examples online show a "Stop-and-Go" approach to function calling: stream text, stop, run tool, start a new stream.
*Entities V1* simplifies this with a **Unified Event Loop**.

The SDK manages the state transitions for you. You use a **single iterator** that:

1.  Streams initial thought/content.
2.  Pauses to yield a `ToolCallRequestEvent` when action is needed.
3.  Automatically submits your tool's result back to the model.
4.  Resumes streaming the final answer immediately within the same loop.

---

## Prerequisite

Please read the definition of tool schemas and function calling here:

[tools_definition.md](/docs/tools_definition.md)

---

```python
import os
import json
from projectdavid import Entity
from projectdavid.events import ContentEvent, ToolCallRequestEvent
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Client
client = Entity()

# -----------------------------------------
# Tool Logic & Registry
#
# Tool execution is a runtime responsibility.
# We map tool names (from LLM) to Python functions.
# -----------------------------------------
def get_flight_times(tool_name, arguments):
    """Actual logic to fetch data."""
    print(f"\n[Runtime] Fetching flights for {arguments.get('departure')}...")
    return json.dumps({
        "status": "success",
        "departure": "10:00 AM PST",
        "arrival": "06:30 PM EST",
        "duration": "4h 30m"
    })

TOOL_REGISTRY = {
    "get_flight_times": get_flight_times
}


# ----------------------------------------------------
# 2. Setup: Thread, Message, Run
# ----------------------------------------------------
assistant_id = "plt_ast_9fnJT01VGrK4a9fcNr8z2O"
thread = client.threads.create_thread()

message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content="Please fetch me the flight times between LAX and NYC, JFK",
    assistant_id=assistant_id,
)

run = client.runs.create_run(assistant_id=assistant_id, thread_id=thread.id)

# ----------------------------------------------------
# 3. Unified Stream Setup
# ----------------------------------------------------
stream = client.synchronous_inference_stream
stream.setup(
    user_id=os.getenv("ENTITIES_USER_ID"),
    thread_id=thread.id,
    assistant_id=assistant_id,
    message_id=message.id,
    run_id=run.id,
    api_key=os.getenv("HYPERBOLIC_API_KEY"),
)

print("Stream started...")

# ----------------------------------------------------
# 4. Single Event Loop (Handles Turns 1..N)
# 
# The generator yields events for the ENTIRE interaction.
# It pauses only when it needs YOU to execute a tool.
# ----------------------------------------------------
try:
    for event in stream.stream_events(
        provider="Hyperbolic", 
        model="hyperbolic/deepseek-ai/DeepSeek-V3"
    ):
        
        # A. Handle Standard Content (Thoughts/Answers)
        if isinstance(event, ContentEvent):
            print(event.content, end="", flush=True)

        # B. Handle Tool Requests
        elif isinstance(event, ToolCallRequestEvent):
            print(f"\n[SDK] Tool Call Detected: {event.tool_name}")

            # 1. Lookup function in registry
            handler = TOOL_REGISTRY.get(event.tool_name)

            if handler:
                # 2. Execute. 
                # The SDK pauses here, executes the function, submits the result, 
                # and silenty triggers the model to resume generating.
                event.execute(handler) 
            else:
                print(f"[!] Unknown tool requested: {event.tool_name}")

except Exception as e:
    print(f"\nError during stream: {e}")

print("\n\nDone.")
```

---

## Example Console Output

```text
Stream started...
[SDK] Tool Call Detected: get_flight_times

[Runtime] Fetching flights for LAX...

The flight from **Los Angeles (LAX)** to **New York (JFK)** has the following details:

- **Flight Duration**: 4 hours and 30 minutes
- **Departure Time**: 10:00 AM PST
- **Arrival Time**: 06:30 PM EST

Done.
```

---

## Lifecycle Summary

The *Entities* SDK implements a **Recursive Turn** pattern:

1.  **Bind:** You bind the client (`bind_clients`), giving the stream permission to manage runs internally.
2.  **Detection:** The stream yields a `ToolCallRequestEvent`.
3.  **Execution:** You call `event.execute(handler)`.
    *   The SDK invokes your Python function.
    *   The SDK submits the result to the API.
    *   The SDK *silently* triggers the next generation step.
4.  **Resumption:** The `for` loop continues immediately, now yielding the `ContentEvent` chunks for the final answer.

The developer writes one loop. The SDK handles the round-trips.