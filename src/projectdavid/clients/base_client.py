# projectdavid/clients/base_client.py (OLD VERSION - Reverted)
import os
from typing import Optional

import httpx
from projectdavid_common.utilities.logging_service import LoggingUtility

logging_utility = LoggingUtility()


class BaseAPIClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        write_timeout: float = 30.0,
    ):
        self.base_url = (
            base_url or os.getenv("ENTITIES_BASE_URL", "http://localhost:9000")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("ENTITIES_API_KEY")

        if not self.base_url:
            raise ValueError("Base URL must be provided via param or environment.")

        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["X-API-Key"] = self.api_key
            logging_utility.info("API Key provided and added to headers.")
        else:
            logging_utility.warning(
                "No API Key provided — protected endpoints may fail."
            )

        self.timeout = httpx.Timeout(
            timeout=timeout,
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
        )

        # Client is created WITH the default headers
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )

        logging_utility.info(
            "[BaseAPIClient] Initialized with base_url: %s and timeout config: %s",
            self.base_url,
            self.timeout,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
