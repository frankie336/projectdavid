# tests/test_clients.py

import json
from unittest.mock import patch
import httpx

# Removed sys.path manipulation as package should be installed editably

# Direct import should now work if package is installed correctly
from entities_sdk.clients.users import UsersClient

def test_users_client_mocked_response():
    """
    Tests the UsersClient create_user method with a mocked HTTP response.
    """
    # Initialize the client (base_url and api_key are illustrative)
    client = UsersClient(base_url="http://fake-url.test", api_key="fake-key-123")

    # Define the expected user data in the mocked response
    mock_user_data = {"id": "usr_123", "name": "test_user"}

    # Create a mock request object (needed for the httpx.Response)
    # The URL should match what the client internally calls
    mock_request = httpx.Request("POST", f"{client.base_url}/v1/users")

    # Create the mock httpx.Response object
    mock_response = httpx.Response(
        status_code=200,
        content=json.dumps(mock_user_data), # Ensure content is JSON string
        request=mock_request
    )

    # Use patch to mock the 'post' method of the client's internal httpx client
    # Ensure the path to the object being patched is correct
    # Assuming client.client refers to the httpx.Client or httpx.AsyncClient instance
    with patch.object(client.client, "post", return_value=mock_response) as mock_post:
        # Call the method under test
        created_user = client.create_user(name="test_user")

        # Assertions:
        # 1. Check if the mock 'post' was called correctly
        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        # Check the URL called
        assert call_args[0] == "/v1/users" # Assumes client prepends base_url automatically
        # Check the JSON payload sent
        assert call_kwargs.get("json") == {"name": "test_user"}

        # 2. Check if the returned user object matches the mocked data
        assert created_user is not None
        assert created_user.id == mock_user_data["id"]
        assert created_user.name == mock_user_data["name"]