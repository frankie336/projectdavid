import os
import time
import mimetypes
from typing import Dict, Any, Optional, BinaryIO

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from ..schemas.file_service import FileResponse
from ..services.logging_service import LoggingUtility

load_dotenv()
logging_utility = LoggingUtility()


class FileClient:
    def __init__(self, base_url=os.getenv("BASE_URL"), api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        logging_utility.info("FileClient initialized with base_url: %s", self.base_url)

    def upload_file(self, file_path: str, user_id: str, purpose: str,
                    metadata: Optional[Dict[str, Any]] = None) -> FileResponse:
        """
        Upload a file to the server, following the OpenAI files endpoint style.

        Args:
            file_path: Path to the file to upload.
            user_id: ID of the user uploading the file.
            purpose: Purpose of the file (e.g., "assistants").
            metadata: Additional metadata (optional).

        Returns:
            FileResponse: The response from the server with file metadata.
        """
        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or 'application/octet-stream'

        logging_utility.info("Uploading file: %s with purpose: %s for user: %s", file_path, purpose, user_id)

        try:
            with open(file_path, 'rb') as file_object:
                # Simplified: Only send the required fields as specified in the FileUploadRequest schema
                form_data = {
                    "purpose": purpose,
                    "user_id": user_id
                }

                files = {'file': (filename, file_object, mime_type)}

                response = self.client.post("/v1/uploads", data=form_data, files=files)
                response.raise_for_status()

                file_data = response.json()
                validated_response = FileResponse.model_validate(file_data)
                logging_utility.info("File uploaded successfully with id: %s", validated_response.id)
                return validated_response

        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error occurred while uploading file: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("An error occurred while uploading file: %s", str(e))
            raise

    def upload_file_object(self, file_object: BinaryIO, file_name: str, user_id: str, purpose: str,
                           metadata: Optional[Dict[str, Any]] = None) -> FileResponse:
        """
        Upload a file-like object to the server.

        Args:
            file_object: File-like object to upload.
            file_name: Name to give the file.
            user_id: ID of the user uploading the file.
            purpose: Purpose of the file.
            metadata: Additional metadata (optional).

        Returns:
            FileResponse: The response from the server with file metadata.
        """
        mime_type, _ = mimetypes.guess_type(file_name)
        mime_type = mime_type or 'application/octet-stream'

        logging_utility.info("Uploading file object: %s with purpose: %s for user: %s", file_name, purpose, user_id)

        try:
            # Simplified: Only send the required fields as specified in the FileUploadRequest schema
            form_data = {
                "purpose": purpose,
                "user_id": user_id
            }

            files = {'file': (file_name, file_object, mime_type)}

            response = self.client.post("/v1/uploads", data=form_data, files=files)
            response.raise_for_status()

            file_data = response.json()
            validated_response = FileResponse.model_validate(file_data)
            logging_utility.info("File uploaded successfully with id: %s", validated_response.id)
            return validated_response

        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error occurred while uploading file: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("An error occurred while uploading file: %s", str(e))
            raise

    def get_file_by_id(self, file_id: str) -> FileResponse:
        """
        Retrieve file metadata by ID.

        Args:
            file_id: The ID of the file to retrieve.

        Returns:
            FileResponse: The file metadata from the server.

        Raises:
            httpx.HTTPStatusError: If HTTP error occurs
            ValueError: If validation error occurs
            Exception: For other errors
        """
        logging_utility.info("Retrieving file with ID: %s", file_id)

        try:
            response = self.client.get(f"/v1/uploads/{file_id}")
            response.raise_for_status()

            file_data = response.json()
            validated_response = FileResponse.model_validate(file_data)
            logging_utility.info("File metadata retrieved successfully for ID: %s", file_id)
            return validated_response

        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error("HTTP error occurred while retrieving file: %s", str(e))
            raise
        except Exception as e:
            logging_utility.error("An error occurred while retrieving file: %s", str(e))
            raise