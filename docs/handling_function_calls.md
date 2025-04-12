# Function Calling

## Overview

Most examples online only show a partial picture of function calling.
They walk through the basics—but skip over what really matters: how to handle, stream, and scale function calls inside a sophisticated state machine like Entities V1.

The script below has been battle-tested in live systems. It's ready for production.
Feel free to adapt it to suit your own assistant workflows.



**Prerequisite** 

Please read here on function call definition:

[function_call_definition.md](/docs/function_call_definition.md)



```python

import os
import json
import time
from projectdavid import Entity
from dotenv import load_dotenv

load_dotenv()

client = Entity()

#-----------------------------------------
# This is a basis mock handler returning 
# static test data. Handling is a consumer 
# side concern. 
#----------------------------------------
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
# Please be aware: 
# - user id needs to be a user id you have generated
# - We are using the default assistant since it is 
# - already highly optimized for function calling.
#-------------------------------------------------------
user_id = "user_oKwebKcvx95018NPtzTaGB"
assistant_id = "default"


#----------------------------------------------------
# Create a thread 
#----------------------------------------------------

thread = client.threads.create_thread(participant_ids=[user_id])

#----------------------------------------------------
# Create a message that should trigger the function call 
#----------------------------------------------------
message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content="Please fetch me the flight times between LAX and NYC, JFK",
    assistant_id=assistant_id,
)

#----------------------------------------------------
# Create a Run 
#----------------------------------------------------

run = client.runs.create_run(
    assistant_id=assistant_id,
    thread_id=thread.id
)


#----------------------------------------------------
# Set up inference.
# - Note: that I am fetching the hyperbolic 
# API key from .env 
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

# --- Stream initial LLM response ---
for chunk in sync_stream.stream_chunks(
    provider="Hyperbolic",
    model="hyperbolic/deepseek-ai/DeepSeek-V3",
    timeout_per_chunk=15.0,
    api_key=os.getenv("HYPERBOLIC_API_KEY"),
):
    content = chunk.get("content", "")
    if content:
        print(content, end="", flush=True)

# --- Function call execution ---
try:
    
    #----------------------------------------------------
    # This is the function call event handler 
    # - Note: that you can tweak timeout & interval 
    # - Alwauys place it here in the order of procedure 
    # ----------------------------------------------------
    
    
    #----------------------------------------------------
    #  This is a special case block. 
    #  Some of the models need a follow-up message before  
    #  they provide you with their synthesis on function call
    #  output. hyperbolic/deepseek-ai/DeepSeek-V3 is an
    #  example of a model we were able to make stable by 
    #  using this method where there is no official 
    #  work around from @DeepSeek
    # ----------------------------------------------------
    
    
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

**The following is real output from the execution above:**

This is the call that the assistant makes. You don't have to deal
with this , but it is included for illustration purposes.
You may need some strategy to filter it out of frontend rendering. 

```bash

{
  "name": "get_flight_times",
  "arguments": {
    "departure": "LAX",
    "arrival": "JFK"
  }
}


```

**Ths is the assistants response:**

```bash

[Tool executed. Generating final response...]

The flight from **Los Angeles (LAX)** to **New York (JFK)** has the following details:

- **Flight Duration**: 4 hours and 30 minutes
- **Departure Time**: 10:00 AM PST
- **Arrival Time**: 06:30 PM EST

Let me know if you'd like additional details or assistance!


```

**Whilst initial set up appears arduous at first, Entities will now
handle the complete life cycle of each and every instance of this call
trigger. You can scale an unlimited number of further functions across
an unlimited number of assistants.**

