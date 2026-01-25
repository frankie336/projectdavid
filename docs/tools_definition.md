```md
# Tools

## Overview

**Tools** allow an assistant to request the execution of external capabilities owned and controlled by the host system.
They enable dynamic, computed, or side-effecting behavior in response to user inputs.

In LLM engineering, **function calling** is the mechanism by which the model *proposes* invoking a tool.
**Tool execution** is performed by the runtime, not by the model.

Function calling is therefore a foundational building block of what are often referred to as *agentic flows*,
but it must not be conflated with tool execution itself.

---

## Defining a Tool (Function Schema)

A tool is defined by exposing a **function schema** to the assistant.
This schema describes:
- the tool’s name
- its purpose
- its expected parameters

The model may emit a **function call** matching this schema, but it does not execute the function.

```python
from projectdavid import Entity

client = Entity()

#---------------------------------
# Install & Import the projectdavid
# SDK.
#---------------------------------

from projectdavid import Entity
client = Entity()

#-------------------------------------
# Create a user
#-------------------------------------

user = client.users.create_user(name='test_user336')
print(f"Created user {user.id}")

# Created user user_oKwebKcvx95018NPtzTaGB

#-------------------------------------
# Create an assistant
#-------------------------------------

assistant = client.assistants.create_assistant(
    name='test_assistant',
    instructions='You are a helpful assistant working at an airport.'
)
print(f"created assistant {assistant.id}")

# created assistant asst_XXPNWcoSEqDOFNuvLLv9vJ

#-------------------------------------------------------
# Define a tool by providing a function schema
#--------------------------------------------------------

assistant = client.assistants.create_assistant(
    name="Flighty",
    instructions="You are a helpful flight attendant",
    model="gpt-oss-120b",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_flight_times",
                "description": "Get flight times",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "departure": {"type": "string"},
                        "arrival": {"type": "string"}
                    },
                    "required": ["departure", "arrival"]
                }
            }
        }
    ]
)
```

---

## Updating an Assistant’s Tools

Tool schemas can be updated or replaced at any time.
This affects which **function calls the model is allowed to emit**.

```python
# -------------------------------------------
# Update an assistant's tools
# --------------------------------------------

update_assistant = client.assistants.update_assistant(
    assistant_id="your_assistant_id",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_flight_times",
                "description": "Get flight times",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "departure": {"type": "string"},
                        "arrival": {"type": "string"}
                    },
                    "required": ["departure", "arrival"]
                }
            }
        }
    ]
)
```

---

## Important Clarification

At this point:

- The **function schema** is attached to the assistant
- The model may now emit **function call outputs** matching this schema

However:

- The model does **not** execute the function
- Additional code is still required to:
  - detect function call outputs
  - validate arguments
  - execute the corresponding tool
  - return results to the model if needed

Please see [`handling_function_calls.md`](/docs/handling_function_calls.md) for a ratified
example of safely handling and executing model-proposed function calls.

---
```
