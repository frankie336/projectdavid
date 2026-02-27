# projectdavid/clients/batfish_client.py
"""
BatfishClient — Tenant-Isolated SDK
=====================================

The server derives tenant isolation from the API key automatically.
Callers never see or handle snapshot_key — they only use snapshot_name.

Usage:
    client = BatfishClient(base_url="http://localhost:9000", api_key="sk_...")

    # 1. Ingest + load snapshot (creates or refreshes)
    client.refresh_snapshot("incident_001")

    # 2. Single RCA tool call (LLM function-call style)
    result = client.run_tool("incident_001", "get_bgp_failures")

    # 3. All tools concurrently
    results = client.run_all_tools("incident_001")

    # 4. List your snapshots
    snapshots = client.list_snapshots()
"""

from typing import Any, Dict, List, Optional

import httpx
from projectdavid_common import UtilsInterface

from projectdavid.clients.base_client import BaseAPIClient

logging_utility = UtilsInterface.LoggingUtility()


class BatfishClient(BaseAPIClient):

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(base_url=base_url, api_key=api_key)
        logging_utility.info("BatfishClient initialised → %s", self.base_url)

    # ── Snapshot management ───────────────────────────────────────────────────

    def refresh_snapshot(
        self,
        snapshot_name: str,
        configs_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest configs recursively and load snapshot into Batfish.
        Creates the snapshot record if new, refreshes if it exists.
        Tenant isolation is derived server-side from the API key.

        Args:
            snapshot_name: Incident/tenant label e.g. "incident_001"
            configs_root:  Override config root path (server-side path, optional)
        """
        params = {"snapshot_name": snapshot_name}
        if configs_root:
            params["configs_root"] = configs_root
        try:
            r = self.client.post("/batfish/snapshot/refresh", params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "refresh_snapshot HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all snapshots owned by the authenticated caller."""
        try:
            r = self.client.get("/batfish/snapshots")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "list_snapshots HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    def get_snapshot(self, snapshot_name: str) -> Dict[str, Any]:
        """Get a single snapshot record by name."""
        try:
            r = self.client.get(f"/batfish/snapshot/{snapshot_name}")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {}
            logging_utility.error(
                "get_snapshot HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    def delete_snapshot(self, snapshot_name: str) -> bool:
        """Soft-delete a snapshot."""
        try:
            r = self.client.delete(f"/batfish/snapshot/{snapshot_name}")
            r.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "delete_snapshot HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    # ── Tool calls ────────────────────────────────────────────────────────────

    def run_tool(self, snapshot_name: str, tool_name: str) -> Dict[str, Any]:
        """
        Run a single named RCA tool against a loaded snapshot.
        This is what the LLM agent calls per function call.

        Args:
            snapshot_name: Snapshot to query
            tool_name:     One of the 8 RCA tools e.g. "get_bgp_failures"
        """
        try:
            r = self.client.post(
                f"/batfish/tool/{tool_name}",
                params={"snapshot_name": snapshot_name},
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

    def run_all_tools(self, snapshot_name: str) -> Dict[str, Any]:
        """
        Run all RCA tools concurrently server-side.
        Returns dict keyed by tool name.
        """
        try:
            r = self.client.post(
                "/batfish/tools/all",
                params={"snapshot_name": snapshot_name},
            )
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
            r = self.client.get("/batfish/tools")
            r.raise_for_status()
            return r.json()["tools"]
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "list_tools HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise

    # ── Health ────────────────────────────────────────────────────────────────

    def check_health(self) -> Dict[str, Any]:
        """Check if the Batfish backend is reachable."""
        try:
            r = self.client.get("/batfish/health")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "check_health HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise
