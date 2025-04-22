# projectdavid/clients/base_client.py
"""
BaseAPIClient

Shared HTTP client that all project‑david service wrappers inherit from.
The old version forced the header `Content‑Type: application/json` on every
request, which broke multipart uploads.  That header has been removed—httpx
will now pick the correct Content‑Type automatically based on the arguments
you pass (`json=`, `data=`, `files=`, etc.).
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
from projectdavid_common import UtilsInterface

logging_utility = UtilsInterface.LoggingUtility()


class BaseAPIClient:
    """Common httpx client with timeout & optional X‑API‑Key handling."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        write_timeout: float = 30.0,
    ) -> None:
        # ------------------------------------------------------------------
        # Resolve configuration
        # ------------------------------------------------------------------
        self.base_url = (
            base_url or os.getenv("ENTITIES_BASE_URL", "http://localhost:9000")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("ENTITIES_API_KEY")

        if not self.base_url:
            raise ValueError(
                "Base URL must be provided via parameter or environment variable."
            )

        # ------------------------------------------------------------------
        # Build default headers
        # NOTE: we do *not* set Content‑Type here—httpx will decide per request.
        # ------------------------------------------------------------------
        headers: dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            logging_utility.info("API Key provided and added to headers.")
        else:
            logging_utility.warning(
                "No API Key provided — protected endpoints may fail."
            )

        # ------------------------------------------------------------------
        # Timeout configuration
        # ------------------------------------------------------------------
        self.timeout = httpx.Timeout(
            timeout=timeout,
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
        )

        # ------------------------------------------------------------------
        # Underlying httpx client
        # ------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------------
    def close(self) -> None:  # pragma: no cover
        """Explicitly close the underlying httpx.Client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
