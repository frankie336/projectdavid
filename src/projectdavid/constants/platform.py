import httpx

DEFAULT_TIMEOUT = httpx.Timeout(timeout=60.0, connect=10.0, read=30.0, write=30.0)
