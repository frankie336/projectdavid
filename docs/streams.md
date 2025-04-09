# Streams

# Entities v1 – Streaming Protocol Reference (`streams.md`)

This document outlines the real-time streaming and Server-Sent Events (SSE) architecture used in Entities v1. It is intended for frontend or SDK developers who need to consume, render, or respond to streamed assistant output.


![Network Diagram](../images/streams.png)

---

## Stream Overview

The assistant streams structured JSON messages, each wrapped in the SSE format:

```
data: {json_payload}
```

Each payload has a `type` field to distinguish the message stream:

```json
{
  "type": "stream",
  "stream_type": "reasoning",  // or: content, code_interpreter, hot_code, final_output
  "content": "..."
}
```

---

## Supported Stream Types

| Stream Type         | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `reasoning`         | Token-by-token inner monologue — never shown to user directly.             |
| `content`           | Natural language intended for the user. This is the default message stream. |
| `hot_code`          | Code the model is actively "thinking about" but hasn't executed yet.        |
| `code_interpreter`  | Outputs from executed Python code blocks (stdout, logs).                    |
| `code_interpreter_stream` | Used when streaming binary content from code execution, e.g. base64 images |
| `tool_call`         | Structured call to a tool (function call).                                 |
| `function_result`   | Response from a tool, streamed back to the assistant.                      |
| `final_output`      | Completion signal: contains the final answer, or a summary.                |

---

## Sample SSE Stream

```
data: {"type": "stream", "stream_type": "reasoning", "content": "Let me think..."}
data: {"type": "stream", "stream_type": "content", "content": "Sure, here's the answer."}
data: {"type": "stream", "stream_type": "code_interpreter", "content": ">>> print('hello')"}
data: {"type": "stream", "stream_type": "code_interpreter_stream", "mime_type": "image/png", "data": "iVBORw0KGgoAAAANS..."}
data: {"type": "stream", "stream_type": "final_output", "content": "✅ Done"}
```

---

## Filtering Stream Chunks

Frontend consumers should filter each `data:` payload based on `stream_type` to determine rendering behavior.

```ts
switch (stream.stream_type) {
  case 'reasoning':
    updateDebugPanel(stream.content);
    break;
  case 'content':
    appendToChat(stream.content);
    break;
  case 'code_interpreter':
    appendToTerminal(stream.content);
    break;
  case 'code_interpreter_stream':
    const blob = base64ToBlob(stream.data, stream.mime_type);
    renderInlinePreview(blob);
    break;
  case 'final_output':
    markRunComplete();
    break;
}
```

---

## Real-World Tips

- `reasoning` often comes before `content`, and may appear redundant.
- `code_interpreter_stream` may include multiple base64 fragments; assemble with care.
- Always sanitize `hot_code` before showing, or use a toggled developer mode.
- Stream order is not guaranteed — always buffer per type if you need strict rendering control.

---

## Contribution

This protocol evolves. If you are adding new stream types or adjusting the payload shape, **document the changes here** and notify the SDK maintainers.

