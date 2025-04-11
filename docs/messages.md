# Messages

## Overview
The Ollama Entities library’s Messages endpoint allows you to create, retrieve, and update messages within conversations. With this endpoint, you can:

Create new messages with specified content, role (e.g., user or assistant), and sender ID.
Retrieve existing message information by ID.
Update message details, such as content or status.
By leveraging the Messages endpoint, you can manage the flow of conversation and interaction between users and assistants, enabling more sophisticated and context-aware conversations.


**Create a Message**

```python
from projectdavid import Entity

client = Entity()

message = client.messages.create_message(
    thread_id='some_thread_id',
    assistant_id='some_assistant_id',
    content="Tell me about current trends in AI",
    role='user'
)

```


**Retrieve a Message**
```python

message = client.message_service.retrieve_message(message_id=message.id)


```


**Update a Message**
```python

message = client.message_service.update_message(message_id=message.id
                                                content='Can you tell me more?',
                                                role='user',
                                                sender_id=user.id
                                                )


```








