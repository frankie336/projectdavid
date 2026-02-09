# /Code Interpreter 

The code interpreter tool is a secure python script execution environment that the assistant can use for data analysis.

```python

import os
import dotenv

from projectdavid import Entity
load_dotenv()

# ------------------------------------------------------------------
# 0.  SDK init + env
# ------------------------------------------------------------------

client = Entity(

    api_key=os.getenv("ENTITIES_API_KEY"),
)


# ------------------------------------------------------------------
# 1.  Insert the code_interpreter into your assistants tools definition 
#     by using its placeholder definition in the tools array.
#     The full definition is orchestrated at run time. 
# ------------------------------------------------------------------


assistant = client.assistants.create_assistant(
    name="test_assistant",
    model="gpt-oss-120b",
    instructions="You are a helpful AI assistant, your name is Nexa.",
    tools=[
        {"type": "code_interpreter"},
    ]
    ,
)

print(assistant.id)
```


## Streaming

When the tool is triggered, the assistant emits a function call that contains the code it wants to execute. The candidate code is pre-streamed; this provides the opportunity to render this stage to the front end for a real time typing effect.

```
{
  "stream_type": "code_execution",
  "chunk": {
    "type": "hot_code",
    "content": "```python\n"
  }
}
```

Once the code has been executed, text based output is streamed in the following envelope.

```
{
  "stream_type": "code_execution",
  "chunk": {
    "type": "hot_code_output",
    "content": "clean_content"
  }
}
```

The assistant can use code_interpreter to generate files, for example a `.png` representing the output of analysis or a `.docx` from a task to document something. Content containing files is streamed in this envelope:

```
{
  "stream_type": "code_execution",
  "chunk": {
    "type": "code_interpreter_stream",
    "content": {
      "filename": "filename",
      "file_id": "file_id",
      "base64": "b64",
      "mime_type": "mime_type"
    }
  }
}
```
