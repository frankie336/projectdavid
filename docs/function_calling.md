# Function Calling

## Overview

Function calling allows your assistant to interact with defined tools, enabling dynamic and computed responses based on user inputs. This section guides you through defining, associating, updating, and deleting functions within your API, as well as integrating these functions into your projects.
Function calls are the bedrock of what are sometimes referred to as agentic flows.


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
    name='the_assistant',
    instructions='You are a helpful customer services agent at an airport'
)

client = Entity()

client.tools.associate_tool_with_assistant(tool_id='some_tool_id',
                                           assistant_id=assistant.id)


```

Your function call is now attached to the assistant. No more code needed.


---

## Function Call handling

Entities provides a scalable, event-driven state management architecture that enables developers to rapidly design, deploy, attach, and manage the full lifecycle of function calls.


### Key Components 

-  ***poll_and_execute_action helper*** 

> The poll_and_execute_action helper is a blocking utility that continuously monitors a run for triggered function calls. It accepts the developer's tool handler as a dependency, automatically executes the logic when a tool is requested, processes the output, and submits the result back to the assistant for response synthesis.




🧱 **Example Usage**
```python
from projectdavid import Entity

client = Entity()



handle_action = client.runs.poll_and_execute_action(
                    run_id=run_id,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    tool_executor=get_flight_times, #<---your call handler here
                    actions_client=client.actions,
                    messages_client=client.messages,
                    timeout=45.0,
                    interval=1.5
                )


```


-  ***Function call Handler*** 


>There is no strict definition for your tool handling logic. Put simply, it can be any computational method that produces output based on the arguments provided by the assistant. Python is  strong function call interface.

🧱 **Example handler**

```python

import json

# -------------------------------------------
# Minimal consumer-defined function call
# handler returning static values
# -------------------------------------------
def get_flight_times(tool_name, arguments):
   
   
    if tool_name == "get_flight_times":
        return json.dumps(
            {
                "status": "success",
                "message": f"Flight from {arguments.get('departure')} to {arguments.get('arrival')}: 4h 30m",
                "departure_time": "10:00 AM PST",
                "arrival_time": "06:30 PM EST",
            }
        )
    return json.dumps(
        {"status": "success", "message": f"Executed tool '{tool_name}' successfully."}
    )

```




📌 **Fully integrated example** 

>The following is a Flask-based integration example that demonstrates how to serve requests and responses to and from an AI assistant built with Entities.



```python

import os
import json
import time
from flask import jsonify, request, Response, stream_with_context
from flask_jwt_extended import jwt_required
from projectdavid import Entity
from projectdavid_common import UtilsInterface

from . import bp_llama

logging_utility = UtilsInterface.LoggingUtility()
client = Entity()


# -------------------------------------------
# Minimal consumer-defined function call
# handler returning static values
# -------------------------------------------
def get_flight_times(tool_name, arguments):
    logging_utility.info(f"[Tool] Executing '{tool_name}' with args: {arguments}")
    time.sleep(1)
    if tool_name == "get_flight_times":
        return json.dumps(
            {
                "status": "success",
                "message": f"Flight from {arguments.get('departure')} to {arguments.get('arrival')}: 4h 30m",
                "departure_time": "10:00 AM PST",
                "arrival_time": "06:30 PM EST",
            }
        )
    return json.dumps(
        {"status": "success", "message": f"Executed tool '{tool_name}' successfully."}
    )


# -------------------------------------------
# Main streaming route for assistant interaction
# -------------------------------------------
@bp_llama.route("/api/messages/process", methods=["POST"])
@jwt_required()
def process_messages():
    if not client:
        logging_utility.error("Client not initialized")
        return jsonify({"error": "Internal server error"}), 500

    logging_utility.info("Request: %s", request.json)
    run_id = thread_id = assistant_id = None

    try:
        # -------------------------------------------
        # Extract and validate payload
        # -------------------------------------------
        data = request.json
        messages = data.get("messages", [])
        user_id = data.get("userId") or data.get("user_id")
        thread_id = data.get("threadId") or data.get("thread_id")
        assistant_id = data.get("assistantId", "default")
        selected_model = data.get("model", "llama3.1")
        provider = data.get("provider", "Hyperbolic")
        api_key = os.getenv("HYPERBOLIC_API_KEY")

        if not thread_id:
            raise ValueError("Missing 'threadId'")
        if not assistant_id:
            raise ValueError("Missing 'assistantId'")
        if not messages:
            raise ValueError("Missing 'messages'")
        user_message_content = messages[-1].get("content", "").strip()
        if not user_message_content:
            raise ValueError("Last message is empty")

        # -------------------------------------------
        # Create message and run
        # -------------------------------------------
        message = client.messages.create_message(
            thread_id=thread_id,
            assistant_id=assistant_id,
            content='Please fetch me the flight time between LAX and JFK, NYC.',
            role="user",
        )
        run = client.runs.create_run(thread_id=thread_id, assistant_id=assistant_id)
        run_id = run.id

        # -------------------------------------------
        # Streaming generator logic
        # -------------------------------------------
        def generate_chunks():
            action_was_handled = False
            logging_utility.info(f"[{run_id}] Starting stream")

            # -------------------------------------------
            # Initial LLM response stream
            # -------------------------------------------
            sync_stream = None
            try:
                if not hasattr(client, "synchronous_inference_stream"):
                    raise AttributeError("Missing stream client")

                sync_stream = client.synchronous_inference_stream
                sync_stream.setup(
                    user_id=user_id,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    message_id=message.id,
                    run_id=run_id,
                    api_key=api_key,
                )

                for chunk in sync_stream.stream_chunks(
                    provider=provider, model=selected_model, api_key=api_key
                ):
                    try:
                        yield json.dumps(chunk) + "\n"
                    except TypeError as te:
                        yield json.dumps(
                            {
                                "type": "error",
                                "error": "Non-serializable chunk",
                                "chunk_repr": repr(chunk),
                            }
                        ) + "\n"
            except Exception as e:
                logging_utility.error(
                    f"[{run_id}] Stream error: {repr(e)}", exc_info=True
                )
                yield json.dumps(
                    {"type": "error", "error": str(e), "run_id": run_id}
                ) + "\n"
                return
            finally:
                if sync_stream and hasattr(sync_stream, "close"):
                    try:
                        sync_stream.close()
                    except Exception as close_err:
                        logging_utility.error(
                            f"[{run_id}] Close error: {repr(close_err)}"
                        )

            # -------------------------------------------
            # If a function call is triggered, it will
            # be handled here
            # -------------------------------------------
            try:

                action_was_handled = client.runs.poll_and_execute_action(
                    run_id=run_id,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    tool_executor=get_flight_times,
                    actions_client=client.actions,
                    messages_client=client.messages,
                    timeout=45.0,
                    interval=1.5,
                )

                if action_was_handled:
                    yield json.dumps(
                        {"type": "status", "status": "tool_execution_complete"}
                    ) + "\n"
            except Exception as err:
                logging_utility.error(f"[{run_id}] Action error: {err}", exc_info=True)
                yield json.dumps(
                    {"type": "error", "error": str(err), "run_id": run_id}
                ) + "\n"

            # -------------------------------------------
            # If a tool was used, stream final response
            # -------------------------------------------
            if action_was_handled:
                yield json.dumps(
                    {"type": "status", "status": "generating_final_response"}
                ) + "\n"

                final_stream = None
                try:
                    final_stream = client.synchronous_inference_stream
                    final_stream.setup(
                        user_id=user_id,
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                        message_id="So, what next?",
                        run_id=run_id,
                        api_key=api_key,
                    )
                    for final_chunk in final_stream.stream_chunks(
                        provider=provider, model=selected_model, api_key=api_key
                    ):
                        try:
                            yield json.dumps(final_chunk) + "\n"
                        except TypeError as te:
                            yield json.dumps(
                                {
                                    "type": "error",
                                    "error": "Non-serializable final chunk",
                                    "chunk_repr": repr(final_chunk),
                                }
                            ) + "\n"
                except Exception as e:
                    logging_utility.error(
                        f"[{run_id}] Final stream error: {repr(e)}", exc_info=True
                    )
                    yield json.dumps(
                        {"type": "error", "error": str(e), "run_id": run_id}
                    ) + "\n"
                finally:
                    if final_stream and hasattr(final_stream, "close"):
                        try:
                            final_stream.close()
                        except Exception as close_err:
                            logging_utility.error(
                                f"[{run_id}] Final close error: {repr(close_err)}"
                            )

            # -------------------------------------------
            # Final status signal
            # -------------------------------------------
            final_status = (
                "tool_completed" if action_was_handled else "inference_complete"
            )
            yield json.dumps(
                {"type": "status", "status": final_status, "run_id": run_id}
            ) + "\n"

        # -------------------------------------------
        # Return stream response
        # -------------------------------------------
        return Response(
            stream_with_context(generate_chunks()),
            content_type="application/x-ndjson",
            headers={
                "X-Conversation-Id": run_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "X-Conversation-Id",
            },
        )

    # -------------------------------------------
    # Exception handling
    # -------------------------------------------
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except AttributeError as ae:
        logging_utility.error("Client misconfigured", exc_info=True)
        return jsonify({"error": "Configuration error"}), 500
    except Exception as e:
        logging_utility.error(f"Unexpected server error: {repr(e)}", exc_info=True)
        return jsonify({"error": "Unexpected error occurred"}), 500

```