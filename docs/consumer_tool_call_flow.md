```text
USER / CLIENT SDK                    DAVID BACKEND API (Worker)                 DATABASE / CACHE
              |                                     |                                     |
    [1]-------|---- POST /process (Turn 1) -------->|                                     |
              |                                     |                                     |
              |                               [2] Inference                               |
              |                               (LLM sees User Msg)                         |
              |                                     |                                     |
              |                               [3] Detects Tool Call                       |
              |                                     |                                     |
              |                               [4] finalize_conversation()---------------->| [SAVE Asst ToolCall]
              |                                     |                                     | [INVALIDATE Context]
    [5]<------|--- Stream Manifest Chunk -----------|                                     |
              |                                     |                                     |
    [6]   (Connection Closed by Server) <-----------| [Worker Returns/Exits]              |
              |                                                                           |
    [7] EXECUTE LOCAL TOOL                                                                |
        (e.g., get_flight_times)                                                          |
              |                                                                           |
    [8]-------|---- POST /submit_tool_output ------>|                                     |
              |                                     |---- [9] Save Tool Result ---------->| [SAVE Tool Output]
    [10]<-----|------- HTTP 200 OK -----------------|                                     |
              |                                                                           |
              |                                                                           |
    [11]------|---- POST /process (Turn 2) -------->|                                     |
              |                                     |                                     |
              |                               [12] _set_up_context()                      |
              |                                    (force_refresh=True)                   |
              |                                     |---- [13] Load full history -------->| [READ ALL MSGS]
              |                                     |<------------------------------------| (User + ToolCall + Result)
              |                                     |                                     |
              |                               [14] Turn 2 Inference                       |
              |                                    (LLM sees tool data)                   |
              |                                     |                                     |
    [15]<-----|--- Stream Final Tokens -------------|                                     |
              |                                     |                                     |
    [16]  (Connection Closed by Server) <-----------| [Worker Returns/Exits]              |
              |                                     |                                     |  

```