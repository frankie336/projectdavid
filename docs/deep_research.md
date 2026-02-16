# Deep Research

Deep Research is a platform capability that orchestrates multiple assistants and tools to perform autonomous, multi-step investigation on a user’s behalf.

When invoked, Deep Research decomposes a user’s query into research tasks and executes a recursive workflow that may include:

- Web search  
- Document retrieval  
- Cross-source comparison  
- Iterative summarisation  
- Synthesis into a structured report  

The system operates agentically, allowing specialised assistants to collaborate in order to:

- Identify relevant sources  
- Validate findings across independent references  
- Refine intermediate hypotheses  
- Produce a final, citation-grounded response aligned with the user’s intent  

Deep Research is designed for queries that require:

- Multi-hop reasoning  
- Evidence gathering  
- Long-horizon synthesis beyond a single model inference pass  

The result is returned as a consolidated research report suitable for downstream reasoning or user presentation.

---

## Deep Research Capability Assignment

You can assign the Deep Research capability to your assistant at creation time.

When enabled, user prompts will be routed into the Deep Research execution loop by default. However, it is recommended that this capability be programmatically toggled on or off depending on the specific use case.

Selective activation allows you to balance latency, cost, and autonomy — ensuring that Deep Research is invoked only when recursive, citation-driven analysis is required, while standard interaction paths remain available for lower-overhead tasks.


```python
from projectdavid import Entity
from dotenv import load_dotenv
# ------------------------------------------------------------------
# 0.  SDK init + env
# ------------------------------------------------------------------
load_dotenv()

client = Entity(
    base_url=os.getenv(),
    api_key=os.getenv("ENTITIES_API_KEY"),
)


assistant = client.assistants.create_assistant(
    name="Test Assistant",
    model="gpt-oss-120b",
    instructions="You are a helpful AI assistant, your name is Nexa.",
    tools=[
        {"type": "web_search"},
    deep_research=True,    
        
        
    ],

)


```
### Toggling states

To toggle between deep research states, use the assistant update method.

```python
update_assistant = client.assistants.update_assistant(
    assistant_id="YOUR_ASSISTANTS_ID",
    deep_research=False,
)
```




---

## Appropriate Use Cases

Use Deep Research when:

- A query requires investigation across multiple sources  
- Evidence must be gathered and validated  
- Intermediate reasoning steps must inform subsequent search  
- A synthesised, citation-based output is required  

Avoid using Deep Research for:

- Simple factual queries  
- Single-step transformations  
- Latency-sensitive conversational turns  
- Tasks that do not require external validation  
