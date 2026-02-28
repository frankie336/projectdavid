# projectdavid/clients/batfish_client.py

from typing import Any, Dict, List, Optional

import httpx
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.batfish_schema import BatfishSnapshotRead

from projectdavid.clients.base_client import BaseAPIClient

logging_utility = UtilsInterface.LoggingUtility()


class BatfishClient(BaseAPIClient):

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(base_url=base_url, api_key=api_key)
        logging_utility.info("BatfishClient initialised → %s", self.base_url)

    # ── SNAPSHOT MANAGEMENT ───────────────────────────────────────────────────

    def create_snapshot(
        self,
        snapshot_name: str,
        configs_root: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> BatfishSnapshotRead:
        """
        Create a new snapshot. Server generates and returns the opaque id.
        Store result.id — use it for all subsequent calls.
        Raises 409 if snapshot_name already exists for this user.
        """
        params = {"snapshot_name": snapshot_name}
        if configs_root:
            params["configs_root"] = configs_root
        if user_id:
            params["user_id"] = user_id

        try:
            r = self.client.post("/v1/batfish/snapshots", params=params)
            r.raise_for_status()
            return BatfishSnapshotRead.model_validate(r.json())
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "create_snapshot HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    def refresh_snapshot(
        self,
        snapshot_id: str,
        configs_root: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> BatfishSnapshotRead:
        """
        Re-ingest configs for an existing snapshot by id.
        Raises 404 if snapshot_id not found for this user.
        """
        params = {}
        if configs_root:
            params["configs_root"] = configs_root
        if user_id:
            params["user_id"] = user_id

        try:
            r = self.client.post(
                f"/v1/batfish/snapshots/{snapshot_id}/refresh", params=params
            )
            r.raise_for_status()
            return BatfishSnapshotRead.model_validate(r.json())
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "refresh_snapshot HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    def get_snapshot(
        self, snapshot_id: str, user_id: Optional[str] = None
    ) -> Optional[BatfishSnapshotRead]:
        """Get a single snapshot record by its opaque id. Returns None if not found."""
        params = {}
        if user_id:
            params["user_id"] = user_id

        try:
            r = self.client.get(f"/v1/batfish/snapshots/{snapshot_id}", params=params)
            r.raise_for_status()
            return BatfishSnapshotRead.model_validate(r.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logging_utility.error(
                "get_snapshot HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    def list_snapshots(
        self, user_id: Optional[str] = None
    ) -> List[BatfishSnapshotRead]:
        """List all snapshots owned by the authenticated caller."""
        params = {}
        if user_id:
            params["user_id"] = user_id

        try:
            r = self.client.get("/v1/batfish/snapshots", params=params)
            r.raise_for_status()
            return [BatfishSnapshotRead.model_validate(s) for s in r.json()]
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "list_snapshots HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    def delete_snapshot(self, snapshot_id: str, user_id: Optional[str] = None) -> bool:
        """Soft-delete a snapshot by its opaque id."""
        params = {}
        if user_id:
            params["user_id"] = user_id

        try:
            r = self.client.delete(
                f"/v1/batfish/snapshots/{snapshot_id}", params=params
            )
            r.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "delete_snapshot HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    # ── TOOL CALLS ────────────────────────────────────────────────────────────

    def run_tool(
        self, snapshot_id: str, tool_name: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a single named RCA tool against a loaded snapshot.
        This is what the LLM agent calls per function call.
        """
        params = {}
        if user_id:
            params["user_id"] = user_id

        try:
            r = self.client.post(
                f"/v1/batfish/snapshots/{snapshot_id}/tools/{tool_name}", params=params
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "run_tool '%s' HTTP %d: %s",
                tool_name,
                e.response.status_code,
                e.response.text,
            )
            raise

    def run_all_tools(
        self, snapshot_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run all 9 RCA tools concurrently server-side.
        Includes get_enriched_topology automatically.
        Returns dict keyed by tool name.
        """
        params = {}
        if user_id:
            params["user_id"] = user_id

        try:
            r = self.client.post(
                f"/v1/batfish/snapshots/{snapshot_id}/tools/all", params=params
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "run_all_tools HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    def get_enriched_topology(
        self, snapshot_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fused L3 topology view — protocol source, MTU, and OSPF/BGP session
        state per subnet. Anomaly-first: healthy subnets suppressed.
        """
        params = {}
        if user_id:
            params["user_id"] = user_id

        try:
            r = self.client.post(
                f"/v1/batfish/snapshots/{snapshot_id}/tools/get_enriched_topology",
                params=params,
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "get_enriched_topology HTTP %d: %s",
                e.response.status_code,
                e.response.text,
            )
            raise

    def list_tools(self) -> List[str]:
        """Return the list of available RCA tool names."""
        try:
            r = self.client.get("/v1/batfish/tools")
            r.raise_for_status()
            return r.json().get("tools", [])
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "list_tools HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    # ── HEALTH ────────────────────────────────────────────────────────────────

    def check_health(self) -> Dict[str, Any]:
        """Check if the Batfish backend is reachable."""
        try:
            r = self.client.get("/v1/batfish/health")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "check_health HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise
