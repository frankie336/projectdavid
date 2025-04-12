# client_sdk/api_keys_client.py (or wherever you keep your SDK clients)

from typing import List, Optional

import httpx
from projectdavid_common.schemas.api_key_schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyDetails,
    ApiKeyListResponse,
)
from projectdavid_common.utilities.logging_service import LoggingUtility

logging_utility = LoggingUtility()


class ApiKeysClient:
    """
    Client for interacting with the API Key management endpoints.
    """

    def __init__(self, base_url: str, api_key: str):
        """
        Initializes the ApiKeysClient.

        Args:
            base_url: The base URL of the Entities API (e.g., "http://localhost:9000").
            api_key: The API key to use for authenticating requests. This key itself
                     must have permissions to manage API keys for the user it belongs to.
        """
        if not base_url:
            raise ValueError("base_url must be provided.")
        if not api_key:
            raise ValueError("api_key must be provided for authentication.")

        self.base_url = base_url.rstrip("/")  # Ensure no trailing slash
        self.api_key = api_key
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}

        # Consider adding timeout configuration
        self.client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=30.0
        )
        logger.info(f"ApiKeysClient initialized for base URL: {self.base_url}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Helper method to make requests and handle common errors."""
        try:
            response = self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()  # Raise HTTPStatusError for 4xx/5xx responses
            return response
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                f"HTTP error occurred: {e.response.status_code} {e.response.reason_phrase} for url {e.request.url}"
            )
            logging_utility.error(f"Response body: {e.response.text}")
            raise  # Re-raise the error for the caller to handle
        except httpx.RequestError as e:
            logging_utility.error(
                f"Request error occurred: {e.__class__.__name__} for url {e.request.url}"
            )
            raise
        except Exception as e:
            logging_utility.error(
                f"An unexpected error occurred: {e.__class__.__name__}"
            )
            raise

    def create_key(
        self,
        user_id: str,
        key_name: Optional[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> ApiKeyCreateResponse:
        """
        Creates a new API key for the specified user.

        Args:
            user_id: The ID of the user for whom to create the key.
            key_name: An optional friendly name for the key.
            expires_in_days: Optional number of days until the key expires.

        Returns:
            An ApiKeyCreateResponse object containing the plain key and details.
            **Store the plain_key securely immediately.**

        Raises:
            httpx.HTTPStatusError: If the API returns an error status (e.g., 401, 403, 404, 500).
            httpx.RequestError: For network or request-related issues.
            pydantic.ValidationError: If the API response doesn't match the expected schema.
            Exception: For other unexpected errors.
        """
        endpoint = f"/v1/users/{user_id}/apikeys"
        logging_utility.info(f"Requesting POST {endpoint}")
        request_data = ApiKeyCreateRequest(
            key_name=key_name, expires_in_days=expires_in_days
        )
        payload = request_data.model_dump(
            exclude_none=True
        )  # Use exclude_none if appropriate

        response = self._make_request("POST", endpoint, json=payload)

        # Validate and return response
        validated_response = ApiKeyCreateResponse.model_validate(response.json())
        logging_utility.info(
            f"API Key created successfully for user {user_id} (Prefix: {validated_response.details.prefix})"
        )
        return validated_response

    def list_keys(
        self, user_id: str, include_inactive: bool = False
    ) -> List[ApiKeyDetails]:
        """
        Lists API keys for the specified user.

        Args:
            user_id: The ID of the user whose keys to list.
            include_inactive: Set to True to include revoked/inactive keys.

        Returns:
            A list of ApiKeyDetails objects. Does not contain the secret keys.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status (e.g., 401, 403, 404, 500).
            httpx.RequestError: For network or request-related issues.
            pydantic.ValidationError: If the API response doesn't match the expected schema.
            Exception: For other unexpected errors.
        """
        endpoint = f"/v1/users/{user_id}/apikeys"
        params = {"include_inactive": include_inactive}
        logging_utility.info(f"Requesting GET {endpoint} with params: {params}")

        response = self._make_request("GET", endpoint, params=params)

        # Validate and return response
        validated_response = ApiKeyListResponse.model_validate(response.json())
        logging_utility.info(
            f"Retrieved {len(validated_response.keys)} API keys for user {user_id}"
        )
        return validated_response.keys

    def get_key_details(self, user_id: str, key_prefix: str) -> ApiKeyDetails:
        """
        Retrieves the details of a specific API key by its prefix.

        Args:
            user_id: The ID of the user who owns the key.
            key_prefix: The unique prefix of the key to retrieve.

        Returns:
            An ApiKeyDetails object for the specified key.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status (e.g., 401, 403, 404, 500).
            httpx.RequestError: For network or request-related issues.
            pydantic.ValidationError: If the API response doesn't match the expected schema.
            Exception: For other unexpected errors.
        """
        endpoint = f"/v1/users/{user_id}/apikeys/{key_prefix}"
        logging_utility.info(f"Requesting GET {endpoint}")

        response = self._make_request("GET", endpoint)

        # Validate and return response
        validated_response = ApiKeyDetails.model_validate(response.json())
        logging_utility.info(
            f"Retrieved details for API key prefix {key_prefix} for user {user_id}"
        )
        return validated_response

    def revoke_key(self, user_id: str, key_prefix: str) -> bool:
        """
        Revokes (deactivates) a specific API key by its prefix.

        Args:
            user_id: The ID of the user who owns the key.
            key_prefix: The unique prefix of the key to revoke.

        Returns:
            True if the key was successfully revoked (API returned 204).

        Raises:
            httpx.HTTPStatusError: If the API returns an error status other than 204
                                   (e.g., 401, 403, 404, 500). A 404 might indicate
                                   the key was already deleted or never existed.
            httpx.RequestError: For network or request-related issues.
            Exception: For other unexpected errors.
        """
        endpoint = f"/v1/users/{user_id}/apikeys/{key_prefix}"
        logging_utility.info(f"Requesting DELETE {endpoint}")

        try:
            # Don't use _make_request directly as we need to check status code 204
            response = self.client.request("DELETE", endpoint)

            if response.status_code == 204:
                logging_utility.info(
                    f"API Key prefix {key_prefix} for user {user_id} revoked successfully."
                )
                return True
            elif response.status_code == 404:
                logging_utility.warning(
                    f"Attempted to revoke key prefix {key_prefix} for user {user_id}, but it was not found."
                )
                # Depending on desired behavior, you might return True or False here,
                # or let raise_for_status handle it below if you want 404 to be an error.
                # Let's treat "not found" as "nothing to revoke" -> successful outcome? Or False?
                # Returning False indicates it wasn't actively revoked *now*.
                return False
            else:
                # Raise HTTPStatusError for other error codes (e.g., 401, 403, 500)
                response.raise_for_status()
                # Should not be reached if raise_for_status works
                logging_utility.error(
                    f"Unexpected status code {response.status_code} during revoke."
                )
                return False  # Should not happen

        except httpx.HTTPStatusError as e:
            logging_utility.error(
                f"HTTP error occurred during revoke: {e.response.status_code} {e.response.reason_phrase} for url {e.request.url}"
            )
            logging_utility.error(f"Response body: {e.response.text}")
            # Re-raise if you want the caller to handle non-204/404 errors specifically
            raise
        except httpx.RequestError as e:
            logging_utility.error(
                f"Request error occurred during revoke: {e.__class__.__name__} for url {e.request.url}"
            )
            raise
        except Exception as e:
            logging_utility.error(
                f"An unexpected error occurred during revoke: {e.__class__.__name__}"
            )
            raise

    def close(self):
        """Closes the underlying HTTP client."""
        if self.client:
            self.client.close()
            logging_utility.info("ApiKeysClient closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
