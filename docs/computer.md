## The /Computer Tool



The computer tool gives the assistant access to a containerised pty session from where it can call linux commands, and interact with the outside world through ad-hock api calls.



### Rooms 
Each pty session is mediated by a room. The assistant must craft, and send its commands to a room. The joining of rooms is completed at the `Entity.threads` level.

```python


# ------------------------------------------------------------------
# 0.  Create thread
# ------------------------------------------------------------------

client = Entity(api_key=os.getenv("ENTITIES_API_KEY"))
thread = client.threads.create_thread()
print(thread.id)

# ------------------------------------------------------------------
# 1.  Join the thread to a pty session
# ------------------------------------------------------------------


session_data = client.computer.create_session(room_id="the_room_id")
        
```

## Monitoring the Computer Tool Session from a Front End

When a computer tool session is created, the platform returns connection metadata that allows a front end application to observe the assistant’s PTY session in real time.

```json
{
  "room_id": "the_room_id",
  "token": "signed_access_token",
  "ws_url": "wss://domain/ws/computer"
}
```

A front end client can use this data to connect to the PTY stream over WebSocket and render terminal output using any terminal emulator component (for example: xterm.js, hterm, or a native terminal widget).

At present, **only the assistant can execute commands** in the PTY. Front ends are observers unless explicitly enabled otherwise by platform policy.

### Connection Flow (Frontend-Agnostic)

1. **Request a session ticket from your backend**
   - Your backend calls:
     - `client.computer.create_session(room_id=...)`
   - The backend returns the signed token and WebSocket URL to the browser.
   - Do not mint tokens directly in the browser.

2. **Open a WebSocket connection**
   - Connect to the provided endpoint using:
     - `ws_url`
     - `room_id`
     - `token`
   - These are typically passed as query parameters or headers depending on your gateway.

3. **Subscribe to computer execution stream events**
   - The server emits JSON messages using a structured stream envelope:

```json
{
  "stream_type": "computer_execution",
  "chunk": {
    "type": "status",
    "status": "complete",
    "run_id": "run_123"
  }
}
```

4. **Interpret chunk payloads**
   - Each message contains:
     - `stream_type`: identifies the stream channel (`computer_execution`)
     - `chunk`: the event payload
   - The `chunk.type` field determines how the message should be handled.

Typical chunk categories may include:

- `status` — lifecycle updates (started, running, complete, failed)
- `output` — terminal stdout/stderr data (ANSI-safe text stream)
- `command` — executed command metadata (optional)

Example status handling:

```text
if chunk.type == "status" and chunk.status == "complete":
    markRunFinished(chunk.run_id)
```

5. **Render terminal output**
   - For output chunks, pass the raw content directly into your terminal renderer.
   - Preserve ANSI escape sequences for correct formatting.
   - Do not reorder chunks — render in arrival order.

6. **Handle lifecycle events**
   Your client should handle:

   - connection open
   - connection close
   - reconnect with backoff
   - room switch cleanup
   - terminal resize events (optional but recommended)
   - run completion via `chunk.type == "status"`

### Minimal Client Responsibilities

Any frontend stack can implement support if it can:

- Open a WebSocket connection
- Parse JSON messages
- Switch logic based on `stream_type` and `chunk.type`
- Render streaming terminal output
- Handle reconnect logic

No framework requirements exist. React, Vue, Svelte, native desktop, or CLI clients are all valid.

### Recommended Safeguards

- Treat tokens as short-lived credentials
- Never expose your platform API key to the browser
- Proxy session creation through your backend
- Validate `room_id` access server-side before issuing a ticket
- Disable stdin in the terminal widget unless interactive mode is explicitly enabled

### Conceptual Architecture

```
Assistant → PTY Session → Room → computer_execution Stream → Frontend Terminal View
```

The frontend is a **viewer of the assistant’s machine**, not the operator — unless you explicitly extend permissions.
