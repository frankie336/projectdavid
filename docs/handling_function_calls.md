# Function Calling and Tool Execution

## Overview

Most examples online only show a partial picture of **function calling**.
They cover schema definition and happy-path demos, but skip what actually matters in production:
how to **detect**, **handle**, **execute**, **stream**, and **scale** model-proposed actions inside a
stateful system such as *Entities V1*.

In LLM engineering:

- **Function calling** is a *model-level capability*  
  (the model emits a structured proposal)
- **Tool execution** is a *runtime responsibility*  
  (the host system validates and executes side effects)

The script below demonstrates the **Event-Driven** pattern. Unlike older polling-based approaches,
this uses a "Smart Iterator" that buffers arguments in real-time and yields an executable
event the moment a tool call is ready.

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

client = Entity()


# -----------------------------------------
# Tool executor (runtime-owned)
#
# This is a mock tool executor. Tool execution
# is a consumer-side concern and is never
# performed by the model itself.
# -----------------------------------------
def get_flight_times(tool_name, arguments):
  if tool_name == "get_flight_times":
    return json.dumps({
      "status": "success",
      "message": f"Flight from {arguments.get('departure')} to {arguments.get('arrival')}: 4h 30m",
      "departure_time": "10:00 AM PST",
      "arrival_time": "06:30 PM EST",
    })
  return json.dumps({"error": f"Unknown tool: {tool_name}"})


assistant_id = "plt_ast_9fnJT01VGrK4a9fcNr8z2O"

# ----------------------------------------------------
# 1. Setup: Thread, Message, Run
# ----------------------------------------------------
thread = client.threads.create_thread()

message = client.messages.create_message(
  thread_id=thread.id,
  role="user",
  content="Please fetch me the flight times between LAX and NYC, JFK",
  assistant_id=assistant_id,
)

run = client.runs.create_run(assistant_id=assistant_id, thread_id=thread.id)

# ----------------------------------------------------
# 2. Initialize the Smart Stream
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

print("Thinking...")

tool_was_executed = False

# ----------------------------------------------------
# 3. Stream Events (The Event-Driven Loop)
#
# Instead of raw chunks, we iterate over high-level
# events. The SDK handles JSON buffering and parsing.
# ----------------------------------------------------
for event in stream.stream_events(
        provider="Hyperbolic",
        model="hyperbolic/deepseek-ai/DeepSeek-V3"
):
  # A. Handle Text Generation
  if isinstance(event, ContentEvent):
    print(event.content, end="", flush=True)

  # B. Handle Tool Requests
  elif isinstance(event, ToolCallRequestEvent):
    print(f"\n[SDK] Tool Call Detected: {event.tool_name}")

    # The 'event' object holds the parsed arguments and 
    # has a helper method to execute and submit the result.
    success = event.execute(get_flight_times)

    if success:
      print("[SDK] Tool executed & result submitted.")
      tool_was_executed = True

# ----------------------------------------------------
# 4. Final Response (If a tool was used)
#
# If a tool was executed, we re-stream to let the
# model generate the final answer using the tool data.
# ----------------------------------------------------
if tool_was_executed:
  print("\n[Generating Final Response...]\n")

  stream.setup(
    user_id=os.getenv("ENTITIES_USER_ID"),
    thread_id=thread.id,
    assistant_id=assistant_id,
    message_id=message.id,
    run_id=run.id,
    api_key=os.getenv("HYPERBOLIC_API_KEY"),
  )

  for event in stream.stream_events(
          provider="Hyperbolic",
          model="hyperbolic/deepseek-ai/DeepSeek-V3"
  ):
    if isinstance(event, ContentEvent):
      print(event.content, end="", flush=True)

print("\n\nDone.")
```

---

## Example Console Output

```text
Thinking...
[SDK] Tool Call Detected: get_flight_times
[SDK] Tool executed & result submitted.

[Generating Final Response...]

The flight from **Los Angeles (LAX)** to **New York (JFK)** has the following details:

- **Flight Duration**: 4 hours and 30 minutes
- **Departure Time**: 10:00 AM PST
- **Arrival Time**: 06:30 PM EST
```

---

## Lifecycle Summary

*Entities* abstracts away the complexity of managing buffers and state transitions:

1.  **Detection:** The SDK monitors the stream. When `call_arguments` arrive, it buffers them internally.
2.  **Event Yield:** Once the tool call is complete and valid, the SDK yields a `ToolCallRequestEvent`.
3.  **Execution:** You call `event.execute(your_function)`. The SDK handles the API round-trip to submit the result.
4.  **Synthesis:** You seamlessly continue streaming the final response.

The model proposes.  
The runtime decides.  
The system remains in control.
```