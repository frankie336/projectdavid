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

    # ── CREATE — new snapshot, 409 if name already exists ────────────────────

    def create_snapshot(
        self,
        snapshot_name: str,
        configs_root: Optional[str] = None,
    ) -> BatfishSnapshotRead:
        """
        Create a new snapshot. Server generates and returns the opaque id.
        Store result.id — use it for all subsequent calls.

        Raises 409 if snapshot_name already exists for this user.
        Use refresh_snapshot(id) to re-ingest an existing snapshot.

        Args:
            snapshot_name: Human label e.g. "incident_001"
            configs_root:  Override config root (server-side path, optional)
        """
        params = {"snapshot_name": snapshot_name}
        if configs_root:
            params["configs_root"] = configs_root
        try:
            r = self.client.post("/v1/batfish/snapshots", params=params)
            r.raise_for_status()
            return BatfishSnapshotRead.model_validate(r.json())
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "create_snapshot HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    # ── REFRESH — re-ingest existing snapshot by id ───────────────────────────

    def refresh_snapshot(
        self,
        snapshot_id: str,
        configs_root: Optional[str] = None,
    ) -> BatfishSnapshotRead:
        """
        Re-ingest configs for an existing snapshot.
        Raises 404 if snapshot_id not found for this user.

        Args:
            snapshot_id:  The id returned by create_snapshot()
            configs_root: Override config root (server-side path, optional)
        """
        params = {}
        if configs_root:
            params["configs_root"] = configs_root
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

    # ── READ / LIST / DELETE ──────────────────────────────────────────────────

    def get_snapshot(self, snapshot_id: str) -> Optional[BatfishSnapshotRead]:
        """Get a single snapshot record by its opaque id. Returns None if not found."""
        try:
            r = self.client.get(f"/v1/batfish/snapshots/{snapshot_id}")
            r.raise_for_status()
            return BatfishSnapshotRead.model_validate(r.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logging_utility.error(
                "get_snapshot HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    def list_snapshots(self) -> List[BatfishSnapshotRead]:
        """List all snapshots owned by the authenticated caller."""
        try:
            r = self.client.get("/v1/batfish/snapshots")
            r.raise_for_status()
            return [BatfishSnapshotRead.model_validate(s) for s in r.json()]
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "list_snapshots HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Soft-delete a snapshot by its opaque id."""
        try:
            r = self.client.delete(f"/v1/batfish/snapshots/{snapshot_id}")
            r.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "delete_snapshot HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    # ── TOOL CALLS ────────────────────────────────────────────────────────────

    def run_tool(self, snapshot_id: str, tool_name: str) -> Dict[str, Any]:
        """
        Run a single named RCA tool against a loaded snapshot.
        This is what the LLM agent calls per function call.

        Args:
            snapshot_id: The id returned by create_snapshot()
            tool_name:   One of the 8 RCA tools e.g. "get_bgp_failures"
        """
        try:
            r = self.client.post(
                f"/v1/batfish/snapshots/{snapshot_id}/tools/{tool_name}"
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

    def run_all_tools(self, snapshot_id: str) -> Dict[str, Any]:
        """Run all RCA tools concurrently server-side. Returns dict keyed by tool name."""
        try:
            r = self.client.post(f"/v1/batfish/snapshots/{snapshot_id}/tools/all")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "run_all_tools HTTP %d: %s", e.response.status_code, e.response.text
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
