import mimetypes
import os
from typing import Any, BinaryIO, Dict, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

from projectdavid.clients.base_client import BaseAPIClient

ent_validator = ValidationInterface()
load_dotenv()
logging_utility = UtilsInterface.LoggingUtility()


class FileClient(BaseAPIClient):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        write_timeout: float = 30.0,
    ):
        """
        FileClient inherits from BaseAPIClient.
        Handles X-API-Key auth and timeout config via shared client logic.
        """
        super().__init__(
            base_url=base_url
            or os.getenv("ENTITIES_BASE_URL", "http://localhost:9000"),
            api_key=api_key or os.getenv("ENTITIES_API_KEY"),
            timeout=timeout,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )
        logging_utility.info("FileClient initialized with base_url: %s", self.base_url)

    def upload_file(
        self,
        file_path: str,
        purpose: str,
        # user_id removed - will be derived from API key on server
        metadata: Optional[
            Dict[str, Any]
        ] = None,  # Metadata not currently used by server endpoint, but keep signature
    ) -> ent_validator.FileResponse:
        """
        Upload a file from a path to the server. Uses authenticated user ID.

        Args:
            file_path (str): Path to the file to upload.
            purpose (str): Purpose of the file (e.g., "assistants").
            metadata (Optional[Dict[str, Any]]): Additional metadata (currently ignored by server).

        Returns:
            FileResponse: The response from the server with file metadata.
        """
        if not os.path.exists(file_path):
            logging_utility.error("File not found at path: %s", file_path)
            raise FileNotFoundError(f"File not found at path: {file_path}")

        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"

        logging_utility.info(
            "Uploading file: %s with purpose: %s",  # Removed user_id from log message context
            file_path,
            purpose,
        )

        try:
            with open(file_path, "rb") as file_object:
                # Only send 'purpose' in form data, user_id is handled by auth
                form_data = {"purpose": purpose}
                # Optional: include metadata if server supports it later
                # if metadata:
                #     form_data['metadata'] = json.dumps(metadata) # Example if metadata needs JSON encoding

                files = {"file": (filename, file_object, mime_type)}

                # Endpoint remains /v1/uploads
                response = self.client.post("/v1/uploads", data=form_data, files=files)
                response.raise_for_status()

                file_data = response.json()
                validated_response = ent_validator.FileResponse.model_validate(
                    file_data
                )
                logging_utility.info(
                    "File uploaded successfully with ID: %s", validated_response.id
                )
                return validated_response

        except ValidationError as e:
            logging_utility.error("Response validation error: %s", e.json())
            # Adapt error message if needed
            raise ValueError(f"Response validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while uploading file '%s': %s - Response: %s",
                filename,
                str(e),
                e.response.text,  # Log response body for 4xx/5xx errors
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An unexpected error occurred while uploading file '%s': %s",
                filename,
                str(e),
                exc_info=True,
            )
            raise

    def upload_file_object(
        self,
        file_object: BinaryIO,
        file_name: str,
        purpose: str,
        # user_id removed - will be derived from API key on server
        metadata: Optional[Dict[str, Any]] = None,  # Keep signature
    ) -> ent_validator.FileResponse:
        """
        Upload a file-like object to the server. Uses authenticated user ID.

        Args:
            file_object (BinaryIO): File-like object to upload.
            file_name (str): Name to assign to the file.
            purpose (str): Purpose of the file.
            metadata (Optional[Dict[str, Any]]): Additional metadata (currently ignored by server).

        Returns:
            FileResponse: The response from the server with file metadata.
        """
        mime_type, _ = mimetypes.guess_type(file_name)
        mime_type = mime_type or "application/octet-stream"

        logging_utility.info(
            "Uploading file object: %s with purpose: %s",  # Removed user_id from log message context
            file_name,
            purpose,
        )

        try:
            # Only send 'purpose' in form data
            form_data = {"purpose": purpose}
            # Optional metadata handling here if needed

            # Ensure file object is read from the beginning if possible
            if file_object.seekable():
                file_object.seek(0)

            files = {"file": (file_name, file_object, mime_type)}

            # Endpoint remains /v1/uploads
            response = self.client.post("/v1/uploads", data=form_data, files=files)
            response.raise_for_status()

            file_data = response.json()
            validated_response = ent_validator.FileResponse.model_validate(file_data)
            logging_utility.info(
                "File object '%s' uploaded successfully with ID: %s",
                file_name,
                validated_response.id,
            )
            return validated_response

        except ValidationError as e:
            logging_utility.error("Response validation error: %s", e.json())
            raise ValueError(f"Response validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error occurred while uploading file object '%s': %s - Response: %s",
                file_name,
                str(e),
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An unexpected error occurred while uploading file object '%s': %s",
                file_name,
                str(e),
                exc_info=True,
            )
            raise

    def retrieve_file(self, file_id: str) -> ent_validator.FileResponse:
        """
        Retrieve file metadata by ID.

        Args:
            file_id (str): The ID of the file to retrieve.

        Returns:
            FileResponse: The file metadata from the server.
        """
        logging_utility.info("Retrieving metadata for file ID: %s", file_id)
        try:
            # *** PATH CHANGED ***
            response = self.client.get(f"/v1/files/{file_id}")
            response.raise_for_status()

            file_data = response.json()
            validated_response = ent_validator.FileResponse.model_validate(file_data)
            logging_utility.info(
                "File metadata retrieved successfully for ID: %s", file_id
            )
            return validated_response

        except ValidationError as e:
            logging_utility.error("Response validation error: %s", e.json())
            raise ValueError(f"Response validation error: {e}")
        except httpx.HTTPStatusError as e:
            # Handle 404 specifically
            if e.response.status_code == 404:
                logging_utility.warning("File not found (404) for ID: %s", file_id)
                raise FileNotFoundError(f"File with ID '{file_id}' not found.") from e
            logging_utility.error(
                "HTTP error occurred while retrieving file metadata for '%s': %s - Response: %s",
                file_id,
                str(e),
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "An unexpected error occurred while retrieving file metadata for '%s': %s",
                file_id,
                str(e),
                exc_info=True,
            )
            raise

    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file by its ID from the server.

        Args:
            file_id (str): The ID of the file to delete.

        Returns:
            bool: True if the file was deleted successfully according to the server response.
        """
        logging_utility.info("Attempting to delete file with ID: %s", file_id)
        try:
            # *** PATH CHANGED ***
            response = self.client.delete(f"/v1/files/{file_id}")
            response.raise_for_status()  # Raises exception for 4xx/5xx errors

            deletion_result = response.json()
            # *** RESPONSE HANDLING CHANGED ***
            # Check the 'deleted' field in the response
            deleted = deletion_result.get("deleted", False)
            if deleted:
                logging_utility.info(
                    "Server confirmed deletion for file ID %s.", file_id
                )
            else:
                logging_utility.warning(
                    "Server response did not confirm deletion for file ID %s: %s",
                    file_id,
                    deletion_result,
                )
            return deleted  # Return the boolean status from the response

        except httpx.HTTPStatusError as e:
            # Handle 404 specifically
            if e.response.status_code == 404:
                logging_utility.warning(
                    "File not found (404) for deletion: %s", file_id
                )
                # You might return False or raise a specific exception like FileNotFoundError
                return False  # Or raise FileNotFoundError(f"File with ID '{file_id}' not found for deletion.") from e
            logging_utility.error(
                "HTTP error occurred while deleting file '%s': %s - Response: %s",
                file_id,
                str(e),
                e.response.text,
            )
            raise  # Re-raise the original error for higher-level handling
        except Exception as e:
            logging_utility.error(
                "An unexpected error occurred while deleting file '%s': %s",
                file_id,
                str(e),
                exc_info=True,
            )
            raise

    def get_signed_url(
        self,
        file_id: str,
        expires_in: Optional[int] = None,
        use_real_filename: Optional[bool] = None,
    ) -> Optional[str]:  # Return Optional[str] as it can fail
        """
        Retrieve a signed URL for the file, optionally specifying expiration and filename usage.

        Args:
            file_id (str): The ID of the file.
            expires_in (Optional[int]): Validity duration in seconds. Server default used if None.
            use_real_filename (Optional[bool]): Request original filename in download URL. Server default used if None.

        Returns:
            Optional[str]: The signed URL, or None if retrieval fails.
        """
        logging_utility.info("Requesting signed URL for file ID: %s", file_id)
        params = {}
        if expires_in is not None:
            params["expires_in"] = expires_in
        if use_real_filename is not None:
            params["use_real_filename"] = use_real_filename

        try:
            # *** PATH CHANGED ***
            response = self.client.get(f"/v1/files/{file_id}/signed-url", params=params)
            response.raise_for_status()
            data = response.json()
            signed_url = data.get("signed_url")
            if signed_url:
                logging_utility.info("Retrieved signed URL for file ID: %s", file_id)
                return signed_url
            else:
                logging_utility.warning(
                    "Signed URL not found in response for file ID: %s", file_id
                )
                return None

        except httpx.HTTPStatusError as e:
            # Handle 404 specifically
            if e.response.status_code == 404:
                logging_utility.warning(
                    "File not found (404) when requesting signed URL for ID: %s",
                    file_id,
                )
                return None  # Or raise FileNotFoundError
            logging_utility.error(
                "HTTP error occurred getting signed URL for '%s': %s - Response: %s",
                file_id,
                str(e),
                e.response.text,
            )
            # Depending on desired behavior, either return None or re-raise
            return None  # Or raise
        except Exception as e:
            logging_utility.error(
                "An unexpected error occurred getting signed URL for '%s': %s",
                file_id,
                str(e),
                exc_info=True,
            )
            return None  # Or raise

    def get_file_as_base64(self, file_id: str) -> Optional[str]:  # Return Optional[str]
        """
        Retrieve the file content as a BASE64-encoded string.

        Args:
            file_id (str): The ID of the file.

        Returns:
            Optional[str]: The BASE64-encoded content, or None if retrieval fails.
        """
        logging_utility.info("Requesting Base64 content for file ID: %s", file_id)
        try:
            # *** PATH CHANGED ***
            response = self.client.get(f"/v1/files/{file_id}/base64")
            response.raise_for_status()
            data = response.json()
            base64_content = data.get("base64")
            if base64_content:
                logging_utility.info(
                    "Retrieved Base64 content for file ID: %s", file_id
                )
                return base64_content
            else:
                logging_utility.warning(
                    "Base64 content not found in response for file ID: %s", file_id
                )
                return None

        except httpx.HTTPStatusError as e:
            # Handle 404 specifically
            if e.response.status_code == 404:
                logging_utility.warning(
                    "File not found (404) when requesting Base64 for ID: %s", file_id
                )
                return None  # Or raise FileNotFoundError
            logging_utility.error(
                "HTTP error occurred getting Base64 for '%s': %s - Response: %s",
                file_id,
                str(e),
                e.response.text,
            )
            return None  # Or raise
        except Exception as e:
            logging_utility.error(
                "An unexpected error occurred getting Base64 for '%s': %s",
                file_id,
                str(e),
                exc_info=True,
            )
            return None  # Or raise
