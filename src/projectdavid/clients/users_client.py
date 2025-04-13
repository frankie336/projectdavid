import os
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

from projectdavid.clients.base_client import BaseAPIClient

# Load environment variables if needed (ensure .env file is present or vars are set)
load_dotenv()

ent_validator = ValidationInterface()
logging_utility = UtilsInterface.LoggingUtility()


class UsersClient(BaseAPIClient):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        write_timeout: float = 10.0,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )
        logging_utility.info("UsersClient initialized with base_url: %s", self.base_url)

    def create_user(self, **kwargs) -> ent_validator.UserRead:
        """
        Creates a new user with the provided details.

        Args:
            **kwargs: Keyword arguments matching the fields in the
                      `ValidationInterface.UserCreate` model (e.g., email,
                      full_name, oauth_provider).

        Returns:
            A UserRead object representing the newly created user.

        Raises:
            ValueError: If the provided data fails validation against UserCreate.
            httpx.HTTPStatusError: If the API returns an error status code.
            Exception: For other network or unexpected errors.
        """
        logging_utility.info("Attempting to create user with provided details.")
        logging_utility.debug(f"User creation raw data: {kwargs}")

        # 1. Validate input data using the UserCreate Pydantic model
        try:
            user_data = ent_validator.UserCreate(**kwargs)
            logging_utility.debug(
                f"Validated user creation data: {user_data.model_dump(exclude_unset=True)}"
            )
        except ValidationError as e:
            logging_utility.error(
                "Validation error for user creation input: %s", e.json()
            )
            raise ValueError(f"Invalid user data provided: {e}") from e

        # 2. Make the API request
        try:
            response = self.client.post(
                "/v1/users", json=user_data.model_dump()
            )  # Use validated data
            response.raise_for_status()  # Raise exception for 4xx/5xx responses
            created_user_json = response.json()

            # 3. Validate the response from the server
            validated_user = ent_validator.UserRead(**created_user_json)
            logging_utility.info(
                "User created successfully with id: %s", validated_user.id
            )
            return validated_user

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:  # Try to parse JSON detail from error response
                error_detail = e.response.json()
            except:
                pass
            logging_utility.error(
                f"HTTP error occurred while creating user ({e.response.status_code}): {error_detail}"
            )
            raise  # Re-raise the original exception
        except ValidationError as e:  # Error validating the server's RESPONSE
            logging_utility.error(
                f"Server response validation error for created user: {e.json()}"
            )
            raise ValueError(f"Invalid user data received from server: {e}") from e
        except Exception as e:
            logging_utility.error(
                f"An unexpected error occurred while creating user: {e}", exc_info=True
            )
            raise RuntimeError(
                f"An unexpected error occurred during user creation: {e}"
            ) from e

    def retrieve_user(self, user_id: str) -> ent_validator.UserRead:
        """Retrieves a specific user by their ID."""
        logging_utility.info("Retrieving user with id: %s", user_id)
        if not user_id:
            raise ValueError("user_id cannot be empty.")
        try:
            response = self.client.get(f"/v1/users/{user_id}")
            response.raise_for_status()
            user_json = response.json()
            validated_user = ent_validator.UserRead(**user_json)
            logging_utility.info("User retrieved successfully: %s", user_id)
            return validated_user
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json()
            except:
                pass
            logging_utility.error(
                f"HTTP error occurred while retrieving user {user_id} ({e.response.status_code}): {error_detail}"
            )
            raise e
        except ValidationError as e:
            logging_utility.error(
                f"Server response validation error for retrieved user {user_id}: {e.json()}"
            )
            raise ValueError(f"Invalid user data received from server: {e}") from e
        except Exception as e:
            logging_utility.error(
                f"An error occurred while retrieving user {user_id}: {e}", exc_info=True
            )
            raise RuntimeError(
                f"An unexpected error occurred retrieving user {user_id}: {e}"
            ) from e

    def update_user(self, user_id: str, **updates) -> ent_validator.UserRead:
        """
        Updates an existing user with the provided details.

        Note: This performs a PUT request. Ensure the API endpoint handles partial
              updates correctly if only partial data is sent via `UserUpdate`.
              The current implementation sends only the fields present in `updates`
              that are valid according to `UserUpdate`.

        Args:
            user_id: The ID of the user to update.
            **updates: Keyword arguments matching fields in `UserUpdate` model.

        Returns:
            A UserRead object representing the updated user.

        Raises:
            ValueError: If the update data fails validation or user_id is missing.
            httpx.HTTPStatusError: If the API returns an error status code.
            Exception: For other network or unexpected errors.
        """
        logging_utility.info("Attempting to update user with id: %s", user_id)
        if not user_id:
            raise ValueError("user_id cannot be empty.")
        if not updates:
            logging_utility.warning(
                "Update user called with no update data for user %s.", user_id
            )
            # Decide whether to proceed (maybe just retrieve) or raise error
            # For now, let's raise, as an update call without updates is ambiguous
            raise ValueError("No update data provided.")

        logging_utility.debug(f"User update raw data for {user_id}: {updates}")

        # 1. Validate the partial update data using UserUpdate
        try:
            validated_update_data = ent_validator.UserUpdate(**updates)
            update_payload = validated_update_data.model_dump(
                exclude_unset=True
            )  # Send only provided fields
            if not update_payload:  # If after validation, no valid fields remain
                raise ValueError("No valid update fields provided after validation.")
            logging_utility.debug(
                f"Validated user update payload for {user_id}: {update_payload}"
            )

        except ValidationError as e:
            logging_utility.error(
                "Validation error for user update input: %s", e.json()
            )
            raise ValueError(
                f"Invalid update data provided for user {user_id}: {e}"
            ) from e

        # 2. Make the API request
        try:
            response = self.client.put(f"/v1/users/{user_id}", json=update_payload)
            response.raise_for_status()
            updated_user_json = response.json()

            # 3. Validate the response
            validated_user = ent_validator.UserRead(**updated_user_json)
            logging_utility.info("User updated successfully: %s", user_id)
            return validated_user

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json()
            except:
                pass
            logging_utility.error(
                f"HTTP error occurred while updating user {user_id} ({e.response.status_code}): {error_detail}"
            )
            raise e
        except ValidationError as e:
            logging_utility.error(
                f"Server response validation error for updated user {user_id}: {e.json()}"
            )
            raise ValueError(
                f"Invalid user data received from server after update: {e}"
            ) from e
        except Exception as e:
            logging_utility.error(
                f"An error occurred while updating user {user_id}: {e}", exc_info=True
            )
            raise RuntimeError(
                f"An unexpected error occurred updating user {user_id}: {e}"
            ) from e

    def delete_user(self, user_id: str) -> ent_validator.UserDeleteResponse:
        """Deletes a specific user by their ID."""
        logging_utility.info("Attempting to delete user with id: %s", user_id)
        if not user_id:
            raise ValueError("user_id cannot be empty.")
        try:
            response = self.client.delete(f"/v1/users/{user_id}")
            response.raise_for_status()
            result_json = response.json()
            # Assuming UserDeleteResponse exists and has fields like { "id": "...", "deleted": true }
            validated_result = ent_validator.UserDeleteResponse(**result_json)
            logging_utility.info("User deleted successfully: %s", user_id)
            return validated_result
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json()
            except:
                pass
            logging_utility.error(
                f"HTTP error occurred while deleting user {user_id} ({e.response.status_code}): {error_detail}"
            )
            raise e
        except ValidationError as e:
            logging_utility.error(
                f"Server response validation error for deleted user {user_id}: {e.json()}"
            )
            raise ValueError(
                f"Invalid delete confirmation received from server: {e}"
            ) from e
        except Exception as e:
            logging_utility.error(
                f"An error occurred while deleting user {user_id}: {e}", exc_info=True
            )
            raise RuntimeError(
                f"An unexpected error occurred deleting user {user_id}: {e}"
            ) from e

    def list_assistants_by_user(
        self, user_id: str
    ) -> List[ent_validator.AssistantRead]:
        """Retrieves a list of assistants associated with a specific user."""
        logging_utility.info("Retrieving assistants for user with id: %s", user_id)
        if not user_id:
            raise ValueError("user_id cannot be empty.")
        try:
            response = self.client.get(f"/v1/users/{user_id}/assistants")
            response.raise_for_status()
            assistants_json = response.json()
            # Ensure the response is a list
            if not isinstance(assistants_json, list):
                logging_utility.error(
                    f"API response for user assistants is not a list: {assistants_json}"
                )
                raise ValueError(
                    "Invalid response format received from server: expected a list."
                )

            validated_assistants = [
                ent_validator.AssistantRead(**assistant)
                for assistant in assistants_json
            ]
            logging_utility.info(
                "Assistants retrieved successfully for user id: %s (%d found)",
                user_id,
                len(validated_assistants),
            )
            return validated_assistants
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json()
            except:
                pass
            logging_utility.error(
                f"HTTP error retrieving assistants for user {user_id} ({e.response.status_code}): {error_detail}"
            )
            raise e
        except ValidationError as e:
            logging_utility.error(
                f"Server response validation error for user assistants {user_id}: {e.json()}"
            )
            raise ValueError(f"Invalid assistant data received from server: {e}") from e
        except Exception as e:
            logging_utility.error(
                f"An error occurred while retrieving assistants for user {user_id}: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"An unexpected error occurred retrieving assistants for {user_id}: {e}"
            ) from e

    def close(self):
        """Closes the underlying HTTP client."""
        if self.client:
            self.client.close()
            logging_utility.info("UsersClient HTTP client closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
