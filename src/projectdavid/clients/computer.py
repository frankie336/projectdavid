# src/projectdavid/clients/computer.py
#
import time
from typing import Any, Dict, Optional

import httpx
from projectdavid_common import (
    UtilsInterface,
)

from projectdavid.clients.base_client import BaseAPIClient

logging_utility = UtilsInterface.LoggingUtility()


class ComputerClientError(Exception):
    """Custom exception for ComputerClient errors."""


class ComputerClient(BaseAPIClient):
    # ------------------------------------------------------------------ #
    #  INIT
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        write_timeout: float = 30.0,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )
        logging_utility.info("ComputerClient ready at: %s", self.base_url)

    # ------------------------------------------------------------------ #
    #  INTERNAL HELPERS (Reuse your pattern)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(response: httpx.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except httpx.DecodingError:
            logging_utility.error("Failed to decode JSON response: %s", response.text)
            raise ComputerClientError("Invalid JSON response from API.")

    def _request_with_retries(self, method: str, url: str, **kwargs) -> httpx.Response:
        retries = 3
        for attempt in range(retries):
            try:
                resp = self.client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                if (
                    exc.response.status_code in {500, 502, 503, 504}
                    and attempt < retries - 1
                ):
                    logging_utility.warning(
                        "ComputerClient: Retrying request (attempt %d)", attempt + 1
                    )
                    time.sleep(2**attempt)
                else:
                    logging_utility.error("ComputerClient Request Failed: %s", exc)
                    raise ComputerClientError(f"API Request Failed: {exc}") from exc
            except httpx.RequestError as exc:
                if attempt < retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise ComputerClientError(f"Network error: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  COMPUTER / SESSION CAPABILITIES
    # ------------------------------------------------------------------ #

    def create_session(self, room_id: str) -> Dict[str, str]:
        """
        Generates a secure, short-lived ticket to join a Computer Room via WebSocket.

        Usage:
            session = client.computer.create_session("room-123")
            # Returns: {'ws_url': '...', 'token': '...', 'room_id': '...'}
        """
        logging_utility.info("Computer: Creating session for Room %s", room_id)

        # For GET requests, we pass data as query parameters, not JSON body
        payload = {"room_id": room_id}

        # CHANGED: "POST" -> "GET" and json=payload -> params=payload
        resp = self._request_with_retries("GET", "/v1/computer/session", params=payload)

        return self._parse_response(resp)
