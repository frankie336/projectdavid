# Function Calling

## Overview

Function calling allows your assistant to interact with defined tools, enabling dynamic and computed responses based on user inputs. This section guides you through defining, associating, updating, and deleting functions within your API, as well as integrating these functions into your projects.


**Define the function**

```python

from projectdavid import Entity

client = Entity()

# Define the function definition
function = {
    "type": "function",
    "function": {
        "name": "get_flight_times",
        "description": "Get the flight times between two cities.",
        "parameters": {
            "type": "object",
            "properties": {
                "departure": {
                    "type": "string",
                    "description": "The departure city (airport code)."
                },
                "arrival": {
                    "type": "string",
                    "description": "The arrival city (airport code)."
                }
            },
            "required": ["departure", "arrival"]
        }
    }
}

tool = client.tools.create_tool(tool_data=function)

print(tool.id)

```


**Attach the tool to the assistant**

**Define the function**

```python

from projectdavid import Entity

client = Entity()

assistant = client.assistants.create_assistant(
    name='movie_db_drone',
    instructions='You are a helpful customer services agent at an airport'
)

client = Entity()

client.tools.associate_tool_with_assistant(tool_id='some_tool_id',
                                           assistant_id=assistant.id)


```

Your function call is now attached to the assistant. No more code needed.


---

## Function Call handling

```python

import json
from projectdavid import Entity, EventsInterface

client = Entity(base_url="http://localhost:9000")


def custom_tool_handler(run_id, run_data, pending_actions):
    for action in pending_actions:
        tool_name = action.get("tool_name")
        args = action.get("function_args", {})
        print(f"Handling tool '{tool_name}' with args {args}")
        # TODO: Add tool logic and optionally submit results:
        # client.runs.submit_tool_outputs(run_id=run_id, outputs=[...])


def main():
    thread_id = "your-thread-id"

    user = client.users.create_user(name='test_user')

    thread = client.threads.create_thread(participant_ids=)

    user_id = "your-user-id"
    user_message = "Your message here"
    assistant_id = "default"
    provider = "Hyperbolic"
    model = "llama3.1"

    # Create a user message
    message = client.messages.create_message(
        thread_id=thread_id,
        assistant_id=assistant_id,
        content=user_message,
        role='user'
    )

    # Initiate a new inference run
    run = client.runs.create_run(thread_id=thread_id, assistant_id=assistant_id)
    print(f"Run initiated: {run.id}")

    # Start monitoring tool invocation events
    monitor = EventsInterface.MonitorLauncher(
        client=client,
        actions_client=client.actions,
        run_id=run.id,
        on_action_required=custom_tool_handler,
        events=EventsInterface
    )
    monitor.start()

    # Setup synchronous inference streaming
    stream = client.synchronous_inference_stream
    stream.setup(
        user_id=user_id,
        thread_id=thread_id,
        assistant_id=assistant_id,
        message_id=message.id,
        run_id=run.id
    )

    # Process and print streamed inference chunks
    try:
        for chunk in stream.stream_chunks(provider=provider, model=model):
            print(json.dumps(chunk, indent=2))
        print("Inference complete.")
    finally:
        stream.close()


if __name__ == '__main__':
    main()

```