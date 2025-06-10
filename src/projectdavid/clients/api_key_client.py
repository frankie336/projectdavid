# client_sdk/api_keys_client.py (or wherever you keep your SDK clients)

from typing import List, Optional

import httpx

# Make sure schemas are correctly imported from your common library
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
    Includes methods for self-service (managing one's own keys)
    and admin actions (managing keys for other users).
    """

    def __init__(self, base_url: str, api_key: str):
        """
        Initializes the ApiKeysClient.

        Args:
            base_url: The base URL of the Entities API (e.g., "http://localhost:9000").
            api_key: The API key to use for authenticating requests.
                     For self-service methods, this key must belong to the user_id being acted upon.
                     For admin methods, this key must belong to a user with admin privileges.
        """
        if not base_url:
            raise ValueError("base_url must be provided.")
        if not api_key:
            # Allow initialization without key for potential public endpoints?
            # For now, assume key is always needed for this client.
            raise ValueError("api_key must be provided for authentication.")

        self.base_url = base_url.rstrip("/")  # Ensure no trailing slash
        self.api_key = api_key
        # Use X-API-Key as per the dependency definition
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}

        # Set a reasonable default timeout
        self.client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=30.0
        )
        logging_utility.info(f"ApiKeysClient initialized for base URL: {self.base_url}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Helper method to make requests and handle common errors."""
        # Ensure the request uses the client's configured API key
        # (httpx.Client automatically uses headers provided during initialization)
        try:
            response = self.client.request(method, endpoint, **kwargs)
            # Raise HTTPStatusError for 4xx/5xx responses, making error handling consistent
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            # Log details for easier debugging
            logging_utility.error(
                f"HTTP error occurred: {e.response.status_code} {e.response.reason_phrase} for url {e.request.url}"
            )
            # Log response body if possible, as it often contains useful error details
            logging_utility.error(f"Response body: {e.response.text}")
            raise e  # Re-raise the original exception
        except httpx.RequestError as e:
            # Handle connection errors, timeouts, etc.
            logging_utility.error(
                f"Request error occurred: {e.__class__.__name__} for url {e.request.url}"
            )
            raise e
        except Exception as e:
            # Catch any other unexpected errors during the request/response cycle
            logging_utility.error(
                f"An unexpected SDK error occurred during request: {e.__class__.__name__}",
                exc_info=True,
            )
            # Wrap in a runtime error or re-raise? Re-raising is often simpler.
            raise e

    # --- Self-Service Methods (Require Auth Key belonging to user_id) ---

    def create_key(
        self,
        user_id: str,  # The user ID must match the user associated with self.api_key
        key_name: Optional[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> ApiKeyCreateResponse:
        """
        [Self-Service] Creates a new API key for the specified user.
        The client must be authenticated with a key belonging to the SAME user_id.

        Args:
            user_id: The ID of the user for whom to create the key (must match authenticated user).
            key_name: An optional friendly name for the key.
            expires_in_days: Optional number of days until the key expires.

        Returns:
            ApiKeyCreateResponse containing the plain key and details.
            **Store the plain_key securely immediately.**
        """
        # Calls the self-service endpoint: /v1/users/{user_id}/apikeys
        endpoint = f"/v1/users/{user_id}/apikeys"
        logging_utility.info(f"[Self-Service] Requesting POST {endpoint}")
        request_data = ApiKeyCreateRequest(
            key_name=key_name, expires_in_days=expires_in_days
        )
        # Send only non-null values if the API expects that
        payload = request_data.model_dump(exclude_none=True)

        response = self._make_request("POST", endpoint, json=payload)

        # Validate API response against the Pydantic schema
        validated_response = ApiKeyCreateResponse.model_validate(response.json())
        logging_utility.info(
            f"[Self-Service] API Key created successfully for user {user_id} (Prefix: {validated_response.details.prefix})"
        )
        return validated_response

    def list_keys(
        self, user_id: str, include_inactive: bool = False
    ) -> List[ApiKeyDetails]:
        """
        [Self-Service] Lists API keys for the specified user.
        The client must be authenticated with a key belonging to the SAME user_id.

        Args:
            user_id: The ID of the user whose keys to list.
            include_inactive: Set to True to include revoked/inactive keys.

        Returns:
            A list of ApiKeyDetails objects. Does not contain the secret keys.
        """
        # Calls the self-service endpoint: /v1/users/{user_id}/apikeys
        endpoint = f"/v1/users/{user_id}/apikeys"
        params = {
            "include_inactive": str(include_inactive).lower()
        }  # Ensure boolean is string for query param
        logging_utility.info(
            f"[Self-Service] Requesting GET {endpoint} with params: {params}"
        )

        response = self._make_request("GET", endpoint, params=params)

        validated_response = ApiKeyListResponse.model_validate(response.json())
        logging_utility.info(
            f"[Self-Service] Retrieved {len(validated_response.keys)} API keys for user {user_id}"
        )
        return validated_response.keys

    def get_key_details(self, user_id: str, key_prefix: str) -> ApiKeyDetails:
        """
        [Self-Service] Retrieves details of a specific API key by prefix.
        The client must be authenticated with a key belonging to the SAME user_id.

        Args:
            user_id: The ID of the user who owns the key.
            key_prefix: The unique prefix of the key to retrieve.

        Returns:
            An ApiKeyDetails object for the specified key.
        """
        # Calls the self-service endpoint: /v1/users/{user_id}/apikeys/{key_prefix}
        endpoint = f"/v1/users/{user_id}/apikeys/{key_prefix}"
        logging_utility.info(f"[Self-Service] Requesting GET {endpoint}")

        response = self._make_request("GET", endpoint)

        validated_response = ApiKeyDetails.model_validate(response.json())
        logging_utility.info(
            f"[Self-Service] Retrieved details for API key prefix {key_prefix} for user {user_id}"
        )
        return validated_response

    def revoke_key(self, user_id: str, key_prefix: str) -> bool:
        """
        [Self-Service] Revokes (deactivates) a specific API key by prefix.
        The client must be authenticated with a key belonging to the SAME user_id.

        Args:
            user_id: The ID of the user who owns the key.
            key_prefix: The unique prefix of the key to revoke.

        Returns:
            True if the key was successfully revoked (API returned 204).
            False if the key was not found (API returned 404).

        Raises:
            httpx.HTTPStatusError: If the API returns an error status other than 204/404.
            httpx.RequestError: For network or request-related issues.
            Exception: For other unexpected errors.
        """
        # Calls the self-service endpoint: /v1/users/{user_id}/apikeys/{key_prefix}
        endpoint = f"/v1/users/{user_id}/apikeys/{key_prefix}"
        logging_utility.info(f"[Self-Service] Requesting DELETE {endpoint}")

        try:
            # Use _make_request but handle specific success/not found codes
            response = self._make_request(
                "DELETE", endpoint
            )  # raise_for_status handles non-2xx

            # If _make_request succeeded (status 2xx), it must be 204 for DELETE success
            if response.status_code == 204:
                logging_utility.info(
                    f"[Self-Service] API Key prefix {key_prefix} for user {user_id} revoked successfully."
                )
                return True
            else:
                # Should ideally not happen if _make_request ensures 2xx on success
                logging_utility.warning(
                    f"[Self-Service] Revoke for {key_prefix} returned unexpected status {response.status_code}"
                )
                return False  # Or raise an error

        except httpx.HTTPStatusError as e:
            # Check if the error was specifically a 404 Not Found
            if e.response.status_code == 404:
                logging_utility.warning(
                    f"[Self-Service] Attempted to revoke key prefix {key_prefix} for user {user_id}, but it was not found (404)."
                )
                return False  # Indicate the key wasn't found to be revoked
            else:
                # Re-raise other HTTP errors (like 401, 403, 500)
                logging_utility.error(
                    f"[Self-Service] HTTP error during revoke: {e.response.status_code}"
                )
                raise e
        except Exception as e:  # Catch other potential errors from _make_request
            logging_utility.error(
                f"[Self-Service] Unexpected SDK error during revoke: {e.__class__.__name__}",
                exc_info=True,
            )
            raise e

    # --- Admin Methods (Require Auth Key belonging to an Admin User) ---

    def create_key_for_user(  # <--- THE NEWLY ADDED METHOD
        self,
        target_user_id: str,  # ID of user to create key for
        key_name: Optional[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> ApiKeyCreateResponse:
        """
        [Admin Action] Creates a new API key for the specified target user.
        Requires the client to be initialized with an **Admin** API key.

        Args:
            target_user_id: The ID of the user for whom to create the key.
            key_name: An optional friendly name for the key.
            expires_in_days: Optional number of days until the key expires.

        Returns:
            An ApiKeyCreateResponse object containing the plain key and details.
            **Store the plain_key securely immediately.**
        """
        # *** Calls the ADMIN-SPECIFIC API ENDPOINT ***
        endpoint = f"/v1/admin/users/{target_user_id}/keys"
        logging_utility.info(
            f"[ADMIN] Requesting POST {endpoint} for target user {target_user_id}"
        )
        request_data = ApiKeyCreateRequest(
            key_name=key_name, expires_in_days=expires_in_days
        )
        payload = request_data.model_dump(exclude_none=True)

        # Use the existing helper to make the request
        response = self._make_request("POST", endpoint, json=payload)

        # Validate and return response using the same schema
        validated_response = ApiKeyCreateResponse.model_validate(response.json())
        logging_utility.info(
            f"[ADMIN] API Key created successfully for user {target_user_id} (Prefix: {validated_response.details.prefix})"
        )
        return validated_response

    # --- Common Methods ---

    def close(self):
        """Closes the underlying HTTP client."""
        if hasattr(self, "client") and self.client:
            self.client.close()
            logging_utility.info("ApiKeysClient closed.")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, ensuring client is closed."""
        self.close()
