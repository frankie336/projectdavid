import time
from typing import Any, Dict, List, Optional

import httpx
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.utilities.logging_service import LoggingUtility

from projectdavid.clients.base_client import BaseAPIClient

# Instantiate the logger
LOG = LoggingUtility()


class EngineerClientError(Exception):
    """Custom exception for EngineerClient interactions."""


class EngineerClient(BaseAPIClient):
    """
    The Client-Side interface for 'The Engineer'.

    This client is responsible for uploading the 'Mental Map' (Inventory)
    from the user's secure network to the Cloud Platform.
    """

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
        LOG.info("EngineerClient ready at: %s", self.base_url)

    # ------------------------------------------------------------------ #
    #  INTERNAL HELPERS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(response: httpx.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except httpx.DecodingError:
            LOG.error("Failed to decode JSON response: %s", response.text)
            raise EngineerClientError("Invalid JSON response from API.")

    def _request_with_retries(self, method: str, url: str, **kwargs) -> httpx.Response:
        retries = 3
        for attempt in range(retries):
            try:
                resp = self.client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                # Retry on Server Errors (5xx)
                if (
                    exc.response.status_code in {500, 502, 503, 504}
                    and attempt < retries - 1
                ):
                    LOG.warning(
                        "EngineerClient: Retrying request due to server error (attempt %d)",
                        attempt + 1,
                    )
                    time.sleep(2**attempt)
                else:
                    # Propagate Client Errors (4xx) or final 5xx
                    LOG.error(
                        "EngineerClient Request Failed: %s %s | Status: %s",
                        method,
                        url,
                        exc.response.status_code,
                    )
                    raise EngineerClientError(f"API Request Failed: {exc}") from exc
            except httpx.RequestError as exc:
                if attempt < retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise EngineerClientError(f"Network error: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  INVENTORY MANAGEMENT (The "Eyes")
    # ------------------------------------------------------------------ #
    def ingest_inventory(
        self,
        assistant_id: str,
        devices: List[Dict[str, Any]],
        clear_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Uploads a list of network devices to build the Assistant's mental map.

        SECURITY NOTE:
        This payload should ONLY contain metadata (Hostname, IP, Platform, Groups).
        Do NOT include passwords or secrets in the 'devices' list.

        Args:
            assistant_id: The UUID of the specific AI Assistant.
            devices: A list of dicts matching the DeviceIngest schema.
                     Example: [{"host_name": "sw1", "platform": "cisco_ios", "groups": ["core"]}]
            clear_existing: If True, wipes previous map for this assistant before adding new ones.

        Returns:
            Dict: Success message and count of devices ingested.
        """
        LOG.info(
            "Engineer: Uploading map for Assistant %s (%d devices)",
            assistant_id,
            len(devices),
        )

        payload = {
            "assistant_id": assistant_id,
            "devices": devices,
            "clear_existing": clear_existing,
        }

        # Route matches 'the_engineer_router.py'
        resp = self._request_with_retries(
            "POST", "/v1/engineer/inventory/ingest", json=payload
        )

        return self._parse_response(resp)
