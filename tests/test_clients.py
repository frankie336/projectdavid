import httpx
from httpx import Response
from unittest.mock import patch

from src.entities_sdk.entities import Entities

def test_users_client_mocked_response():
    client = Entities(base_url="http://fake-url", api_key="key")

    with patch.object(client.users.client, "post", return_value=Response(200, json={"id": "123"})):
        response = client.users.create_user(name="test")
        assert response.status_code == 200
        assert response.json()["id"] == "123"
