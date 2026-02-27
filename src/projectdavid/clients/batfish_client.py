#! projectdavid/clients/batfish_client.py
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface

from projectdavid.clients.base_client import BaseAPIClient

load_dotenv()
logging_utility = UtilsInterface.LoggingUtility()


class BatfishClient(BaseAPIClient):
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(base_url=base_url, api_key=api_key)
        logging_utility.info(
            "BatfishClient initialized with base_url: %s", self.base_url
        )

    def refresh_snapshot(self, incident: str) -> Dict[str, Any]:
        """
        Pull latest GNS3 configs and refresh the Batfish snapshot.
        """
        logging_utility.info("Refreshing Batfish snapshot for incident: %s", incident)
        try:
            # incident is a query parameter in the router
            response = self.client.post(
                "/batfish/snapshot/refresh", params={"incident": incident}
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d error refreshing snapshot: %s",
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error("Unexpected error refreshing snapshot: %s", str(e))
            raise

    def get_rca_prompt(self, incident: str, refresh: bool = True) -> Dict[str, Any]:
        """
        Full pipeline: optionally refresh snapshot, run Batfish queries,
        and return a structured prompt ready for LLM consumption.
        """
        logging_utility.info(
            "Fetching RCA prompt for incident: %s (refresh=%s)", incident, refresh
        )
        try:
            response = self.client.post(
                "/batfish/rca/prompt", params={"incident": incident, "refresh": refresh}
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d error building RCA prompt: %s",
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error("Unexpected error building RCA prompt: %s", str(e))
            raise

    def list_snapshot_devices(self, incident: str) -> Dict[str, Any]:
        """
        List config files currently ingested into a specific snapshot.
        """
        logging_utility.info("Listing snapshot devices for incident: %s", incident)
        try:
            response = self.client.get(f"/batfish/snapshot/{incident}/devices")
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logging_utility.warning("Snapshot '%s' not found.", incident)
                return {"incident": incident, "device_count": 0, "devices": []}

            logging_utility.error(
                "HTTP %d error listing snapshot devices: %s",
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "Unexpected error listing snapshot devices: %s", str(e)
            )
            raise

    def check_health(self) -> Dict[str, Any]:
        """
        Check if the Batfish backend service is reachable.
        """
        logging_utility.info("Checking Batfish service health")
        try:
            response = self.client.get("/batfish/health")
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP %d error checking Batfish health: %s",
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "Unexpected error checking Batfish health: %s", str(e)
            )
            raise
