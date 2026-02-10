## The /Web Search Tool


The web search tool gives the assistant real-time read access to the World Wide Web.
The assistant can directly navigate to pages, read content, perform searches, and synthesise responses using website data as grounding.



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
# 1.  Insert the web_search into your assistants tools definition 
#     by using its placeholder definition in the tools array.
#     The full definition is orchestrated at run time.
#     
#     Set the web_access_parameter to true
# ------------------------------------------------------------------


assistant = client.assistants.create_assistant(
    name="test_assistant",
    model="gpt-oss-120b",
    instructions="You are a helpful AI assistant, your name is Nexa.",
    tools=[
        {"type": "web_search"},
    ],
    web_access = True
    
)
print(assistant.id)
```

### SERP Discovery

The web search tool supports Search Engine Results Page (SERP) discovery.
This is the programmatic exploration and extraction of structured intelligence from search results pages, rather than from a single known URL.
It gives the assistant a versatile, multi-step web research capability grounded in live sources.