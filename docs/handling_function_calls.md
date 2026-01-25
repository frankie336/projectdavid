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

The script below demonstrates a **production-grade action handling loop**.
It has been battle-tested in live systems and is safe to adapt for your own assistant workflows.

---

## Prerequisite

Please read the definition of tool schemas and function calling here:

[tools_definition.md](/docs/tools_definition.md)

---

```python
import os
import json
import time
from projectdavid import Entity
from dotenv import load_dotenv

load_dotenv()

client = Entity()

#-----------------------------------------
# Tool executor (runtime-owned)
#
# This is a mock tool executor that returns
# static test data. Tool execution is a
# consumer-side concern and is never
# performed by the model itself.
#-----------------------------------------
def get_flight_times(tool_name, arguments):
    if tool_name == "get_flight_times":
        return json.dumps({
            "status": "success",
            "message": f"Flight from {arguments.get('departure')} to {arguments.get('arrival')}: 4h 30m",
            "departure_time": "10:00 AM PST",
            "arrival_time": "06:30 PM EST",
        })
    return json.dumps({
        "status": "success",
        "message": f"Executed tool '{tool_name}' successfully."
    })

#------------------------------------------------------
# Notes:
# - user_id must reference an existing user
# - the default assistant is used here because it is
#   already optimized for function calling behavior
#------------------------------------------------------
user_id = "user_oKwebKcvx95018NPtzTaGB"
assistant_id = "default"

#----------------------------------------------------
# Create a thread
#----------------------------------------------------
thread = client.threads.create_thread(participant_ids=[user_id])

#----------------------------------------------------
# Create a user message that may trigger a function call
#----------------------------------------------------
message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content="Please fetch me the flight times between LAX and NYC, JFK",
    assistant_id=assistant_id,
)

#----------------------------------------------------
# Create a run
#----------------------------------------------------
run = client.runs.create_run(
    assistant_id=assistant_id,
    thread_id=thread.id
)

#----------------------------------------------------
# Set up inference
# - API key is sourced from environment variables
#----------------------------------------------------
sync_stream = client.synchronous_inference_stream
sync_stream.setup(
    user_id=user_id,
    thread_id=thread.id,
    assistant_id=assistant_id,
    message_id=message.id,
    run_id=run.id,
    api_key=os.getenv("HYPERBOLIC_API_KEY"),
)

#----------------------------------------------------
# Stream the initial model response
#----------------------------------------------------
for chunk in sync_stream.stream_chunks(
    provider="Hyperbolic",
    model="hyperbolic/deepseek-ai/DeepSeek-V3",
    timeout_per_chunk=15.0,
    api_key=os.getenv("HYPERBOLIC_API_KEY"),
):
    content = chunk.get("content", "")
    if content:
        print(content, end="", flush=True)

#----------------------------------------------------
# Handle model-proposed function calls and execute tools
#----------------------------------------------------
try:
    #----------------------------------------------------
    # Action handling loop
    #
    # This block:
    # - polls for function call outputs from the model
    # - validates them
    # - executes the corresponding tools
    #
    # Tool execution is deterministic and controlled
    # entirely by the runtime.
    #----------------------------------------------------
    action_was_handled = client.runs.poll_and_execute_action(
        run_id=run.id,
        thread_id=thread.id,
        assistant_id=assistant_id,
        tool_executor=get_flight_times,
        actions_client=client.actions,
        messages_client=client.messages,
        timeout=45.0,
        interval=1.5,
    )

    #----------------------------------------------------
    # Some models require a follow-up inference pass
    # after tool execution to synthesize a final response.
    #
    # This pattern stabilizes models such as:
    # hyperbolic/deepseek-ai/DeepSeek-V3
    #----------------------------------------------------
    if action_was_handled:
        print("\n[Tool executed. Generating final response...]\n")

        sync_stream.setup(
            user_id=user_id,
            thread_id=thread.id,
            assistant_id=assistant_id,
            message_id="regenerated",
            run_id=run.id,
            api_key=os.getenv("HYPERBOLIC_API_KEY"),
        )

        for final_chunk in sync_stream.stream_chunks(
            provider="Hyperbolic",
            model="hyperbolic/deepseek-ai/DeepSeek-V3",
            timeout_per_chunk=15.0,
            api_key=os.getenv("HYPERBOLIC_API_KEY"),
        ):
            content = final_chunk.get("content", "")
            if content:
                print(content, end="", flush=True)

except Exception as e:
    print(f"\n[Error during tool execution or final stream]: {str(e)}")
```

---

## Example Function Call Output (Model-Level)

The following is a **function call emitted by the model**.
It is *not executed by the model* and is shown here for illustration purposes only.

You may wish to filter this event from frontend rendering.

```json
{
  "name": "get_flight_times",
  "arguments": {
    "departure": "LAX",
    "arrival": "JFK"
  }
}
```

---

## Example Final Assistant Response

```text
[Tool executed. Generating final response...]

The flight from **Los Angeles (LAX)** to **New York (JFK)** has the following details:

- **Flight Duration**: 4 hours and 30 minutes
- **Departure Time**: 10:00 AM PST
- **Arrival Time**: 06:30 PM EST

Let me know if you'd like additional details or assistance!
```

---

## Lifecycle Summary

While the initial setup may appear involved, *Entities* now manages the complete lifecycle of:

- function call detection
- tool execution
- result injection
- response synthesis

You can safely scale:
- an unlimited number of tools
- across an unlimited number of assistants
- with consistent, auditable behavior

The model proposes.  
The runtime decides.  
The system remains in control.

---
```
