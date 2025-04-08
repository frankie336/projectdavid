from tkinter.font import names

# Assistants

## Overview

Create an Assistant by defining its custom instructions and picking a model. If helpful, add files and enable tools like Code Interpreter, File Search, and Function calling.

### Basic Assistant Operations

```python

from projectdavid import Entity

client = Entity()

# Create assistant

assistant = client.assistants.create_assistant(
    name='Mathy',
    description='test_case',
    model='llama3.1',
    instructions='You are a helpful math tutor.'
)
print(f"Assistant created: ID: {assistant.id}")

# Retrieve an Assistant

retrieved_assistant = client.assistants.retrieve_assistant(assistant_id=assistant.id)

print(retrieved_assistant)

# Update and assistant

client.assistants.update_assistant(
    assistant_id=assistant.id,
    name='Mathy',
    description='test_update',
    instructions='You are now a world class poker player.'
)

# Delete an assistant 

client.assistants.delete_assistant(assistant_id=assistant.id)

```