import sys
from pathlib import Path
import json
from unittest.mock import patch
import httpx

# Add src to path for import to work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entities_sdk.clients.users import UsersClient  # make sure this import works

def test_users_client_mocked_response():
    client = UsersClient(base_url="http://fake-url", api_key="key")

    mock_user = {"id": "123", "name": "test"}
    mock_request = httpx.Request("POST", "http://fake-url/v1/users")

    # Properly simulate the response with content and request
    mock_response = httpx.Response(
        status_code=200,
        content=json.dumps(mock_user),
        request=mock_request
    )

    # Patch the internal HTTPX client
    with patch.object(client.client, "post", return_value=mock_response):
        user = client.create_user(name="test")

        assert user.id == "123"
        assert user.name == "test"
