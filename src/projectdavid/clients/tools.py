import time
from typing import Any, Dict, Optional

import httpx
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.utilities.logging_service import LoggingUtility

from projectdavid.clients.base_client import BaseAPIClient

# [FIX] Instantiate the logger so we don't pass strings as 'self'
LOG = LoggingUtility()


class ToolsClientError(Exception):
    """Custom exception for ToolsClient errors."""


class ToolsClient(BaseAPIClient):
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
        # [FIX] Use the instance (LOG), not the class (LoggingUtility)
        LOG.info("ToolsClient ready at: %s", self.base_url)

    # ------------------------------------------------------------------ #
    #  INTERNAL HELPERS (Matches AssistantsClient Pattern)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(response: httpx.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except httpx.DecodingError:
            # [FIX] Use LOG instance
            LOG.error("Failed to decode JSON response: %s", response.text)
            raise ToolsClientError("Invalid JSON response from API.")

    def _request_with_retries(self, method: str, url: str, **kwargs) -> httpx.Response:
        retries = 3
        for attempt in range(retries):
            try:
                # self.client is inherited from BaseAPIClient (Sync httpx.Client)
                resp = self.client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                # Retry on Server Errors (5xx)
                if (
                    exc.response.status_code in {500, 502, 503, 504}
                    and attempt < retries - 1
                ):
                    # [FIX] Use LOG instance
                    LOG.warning(
                        "ToolsClient: Retrying request due to server error (attempt %d)",
                        attempt + 1,
                    )
                    time.sleep(2**attempt)
                else:
                    # Propagate Client Errors (4xx) or final 5xx
                    # [FIX] Use LOG instance
                    LOG.error(
                        "ToolsClient Request Failed: %s %s | Status: %s",
                        method,
                        url,
                        exc.response.status_code,
                    )
                    raise ToolsClientError(f"API Request Failed: {exc}") from exc
            except httpx.RequestError as exc:
                if attempt < retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise ToolsClientError(f"Network error: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  WEB BROWSING CAPABILITIES
    # ------------------------------------------------------------------ #
    def web_read(self, url: str, force_refresh: bool = False) -> str:
        """
        Orders the API to visit a URL, scrape it, and return the first page (Page 0).

        Args:
            url: The website to visit.
            force_refresh: If True, ignores Redis cache and re-scrapes.

        Returns:
            str: The formatted content of the web page.
        """
        # [FIX] Use LOG instance
        LOG.info("Tools: Reading URL %s (refresh=%s)", url, force_refresh)

        payload = {"url": url, "force_refresh": force_refresh}

        resp = self._request_with_retries("POST", f"/v1/tools/web/read", json=payload)

        data = self._parse_response(resp)
        return data.get("content", "")

    def web_scroll(self, url: str, page: int) -> str:
        """
        Retrieves a specific page chunk from a previously read URL.

        Args:
            url: The website originally read.
            page: The page number to retrieve (0-indexed).

        Returns:
            str: The formatted content of the specific page chunk.
        """
        # [FIX] Use LOG instance
        LOG.info("Tools: Scrolling URL %s to page %d", url, page)

        payload = {"url": url, "page": page}

        resp = self._request_with_retries("POST", f"/v1/tools/web/scroll", json=payload)

        data = self._parse_response(resp)
        return data.get("content", "")

    def web_search(self, url: str, query: str) -> str:
        """
        Searches the entire document (all pages) for a specific keyword or phrase.
        Use this instead of scrolling if you are looking for specific data.

        Args:
            url: The website URL.
            query: The keyword to find (e.g., 'pricing', 'contact', 'API key').

        Returns:
            str: Snippets of text where the query was found, with Page numbers.
        """
        # [FIX] Use LOG instance
        LOG.info("Tools: Searching URL %s for '%s'", url, query)

        payload = {"url": url, "query": query}

        resp = self._request_with_retries("POST", f"/v1/tools/web/search", json=payload)

        data = self._parse_response(resp)
        return data.get("content", "")
