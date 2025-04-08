# Threads

## Overview

A thread represents the conversation dialogue between a user and an assistant, providing a coherent data structure for conversation state management.

Multiple assistants can access the same thread, enabling seamless conversation continuity.
Threads facilitate ready-to-use conversation context, making it easier to manage complex interactions.


**Thread operations**

```python
from projectdavid import Entity

client = Entity()

user = client.users.create_user(name='My test user')

thread = client.threads.create_thread(participant_ids=user.id)

print(thread.id)

user_threads = client.threads.list_threads(user_id='your_user_id')

print(user_threads)

# Delete a thread

client.threads.delete_thread(thread_id=thread.id)

```


