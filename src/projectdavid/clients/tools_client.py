import os
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common import UtilsInterface, ValidationInterface
from pydantic import ValidationError

ent_validator = ValidationInterface()

load_dotenv()

logging_utility = UtilsInterface.LoggingUtility()


class ToolsClient:

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
        ToolsClient inherits from BaseAPIClient.
        Handles API key injection and timeout config using common base.
        """
        super().__init__(
            base_url=base_url
            or os.getenv("ENTITIES_BASE_URL", "http://localhost:9000/"),
            api_key=api_key or os.getenv("ENTITIES_API_KEY"),
            timeout=timeout,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )
        logging_utility.info(
            "ActionsClient initialized with base_url: %s", self.base_url
        )

    def __del__(self):
        # Ensure the client is closed to prevent resource leaks.
        self.client.close()

    def create_tool(self, **tool_data) -> ent_validator.ToolRead:
        logging_utility.info("Creating new tool")
        try:
            tool = ent_validator.ToolCreate(**tool_data)
            response = self.client.post("/v1/tools", json=tool.model_dump())
            response.raise_for_status()
            created_tool = response.json()
            validated_tool = ent_validator.ToolRead.model_validate(created_tool)
            logging_utility.info(
                "Tool created successfully with id: %s", validated_tool.id
            )
            return validated_tool
        except ValidationError as e:
            logging_utility.error("Validation error during tool creation: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error during tool creation: %s | Response: %s",
                str(e),
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error("Unexpected error during tool creation: %s", str(e))
            raise

    def associate_tool_with_assistant(self, tool_id: str, assistant_id: str) -> None:
        logging_utility.info(
            "Associating tool %s with assistant %s", tool_id, assistant_id
        )
        try:
            response = self.client.post(
                f"/v1/assistants/{assistant_id}/tools/{tool_id}"
            )
            response.raise_for_status()
            logging_utility.info(
                "Tool %s associated with assistant %s successfully",
                tool_id,
                assistant_id,
            )
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error during tool-assistant association: %s | Response: %s",
                str(e),
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "Unexpected error during tool-assistant association: %s", str(e)
            )
            raise

    def disassociate_tool_from_assistant(self, tool_id: str, assistant_id: str) -> None:
        logging_utility.info(
            "Disassociating tool %s from assistant %s", tool_id, assistant_id
        )
        try:
            response = self.client.delete(
                f"/v1/assistants/{assistant_id}/tools/{tool_id}"
            )
            response.raise_for_status()
            logging_utility.info(
                "Tool %s disassociated from assistant %s successfully",
                tool_id,
                assistant_id,
            )
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error during tool-assistant disassociation: %s | Response: %s",
                str(e),
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error(
                "Unexpected error during tool-assistant disassociation: %s", str(e)
            )
            raise

    def get_tool_by_id(self, tool_id: str) -> ent_validator.ToolRead:
        """Retrieve a tool by its ID."""
        logging_utility.info("Retrieving tool with id: %s", tool_id)
        try:
            response = self.client.get(f"/v1/tools/{tool_id}")
            response.raise_for_status()
            tool = response.json()
            validated_tool = ent_validator.ToolRead.model_validate(tool)
            logging_utility.info("Tool retrieved successfully")
            return validated_tool
        except ValidationError as e:
            logging_utility.error(
                "Validation error during tool retrieval: %s", e.json()
            )
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error during tool retrieval: %s | Response: %s",
                str(e),
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error("Unexpected error during tool retrieval: %s", str(e))
            raise

    def get_tool_by_name(self, name: str) -> ent_validator.ToolRead:
        """Retrieve a tool by its name."""
        logging_utility.info("Retrieving tool with name: %s", name)
        try:
            response = self.client.get(f"/v1/tools/name/{name}")
            response.raise_for_status()
            tool = response.json()
            validated_tool = ent_validator.ToolRead.model_validate(tool)
            logging_utility.info("Tool retrieved successfully")
            return validated_tool
        except ValidationError as e:
            logging_utility.error(
                "Validation error during tool retrieval: %s", e.json()
            )
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error during tool retrieval: %s | Response: %s",
                str(e),
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error("Unexpected error during tool retrieval: %s", str(e))
            raise

    def update_tool(
        self, tool_id: str, tool_update: ent_validator.ToolUpdate
    ) -> ent_validator.ToolRead:
        logging_utility.info("Updating tool with ID: %s", tool_id)
        try:
            response = self.client.put(
                f"/v1/tools/{tool_id}", json=tool_update.model_dump(exclude_unset=True)
            )
            response.raise_for_status()
            updated_tool = response.json()
            validated_tool = ent_validator.ToolRead.model_validate(updated_tool)
            logging_utility.info("Tool updated successfully with ID: %s", tool_id)
            return validated_tool
        except ValidationError as e:
            logging_utility.error("Validation error during tool update: %s", e.json())
            raise ValueError(f"Validation error: {e}")
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error during tool update: %s | Response: %s",
                str(e),
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error("Unexpected error during tool update: %s", str(e))
            raise

    def delete_tool(self, tool_id: str) -> None:
        logging_utility.info("Deleting tool with id: %s", tool_id)
        try:
            response = self.client.delete(f"/v1/tools/{tool_id}")
            response.raise_for_status()
            logging_utility.info("Tool deleted successfully with ID: %s", tool_id)
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error during tool deletion: %s | Response: %s",
                str(e),
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error("Unexpected error during tool deletion: %s", str(e))
            raise

    def parse_parameters(self, parameters):
        """Recursively parse parameters and handle different structures."""
        if isinstance(parameters, dict):
            parsed = {}
            for key, value in parameters.items():
                if isinstance(value, dict):
                    parsed[key] = self.parse_parameters(value)
                else:
                    parsed[key] = value
            return parsed
        return parameters

    def restructure_tools(self, tools):
        """Restructure the tools to match the target structure."""
        restructured_tools = []
        for tool in tools:
            function_info = tool["function"]
            # The function details might be nested
            if "function" in function_info:
                function_info = function_info["function"]
            restructured_tool = {
                "type": "function",
                "function": {
                    "name": function_info.get("name", "Unnamed tool"),
                    "description": function_info.get(
                        "description", "No description provided"
                    ),
                    "parameters": function_info.get("parameters", {}),
                },
            }
            restructured_tools.append(restructured_tool)
        return restructured_tools

    def list_tools(
        self, assistant_id: Optional[str] = None, restructure: bool = False
    ) -> List[dict]:
        """
        List tools, optionally for a specific assistant and optionally restructure the response.

        Args:
            assistant_id (Optional[str]): The assistant ID to filter tools.
            restructure (bool): Whether to restructure the tools.

        Returns:
            List[dict]: A list of tools.
        """
        url = f"/v1/assistants/{assistant_id}/tools" if assistant_id else "/v1/tools"
        logging_utility.info("Listing tools for assistant ID: %s", assistant_id)
        try:
            response = self.client.get(url)
            response.raise_for_status()
            tools_list = response.json()
            logging_utility.info("Fetched tool list: %s", tools_list)
            tools = tools_list["tools"]
            logging_utility.info("Retrieved %d tools", len(tools))
            if restructure:
                restructured_tools = self.restructure_tools(tools)
                logging_utility.info("Restructured tools: %s", restructured_tools)
                return restructured_tools
            else:
                return tools
        except httpx.HTTPStatusError as e:
            logging_utility.error(
                "HTTP error while listing tools: %s | Response: %s",
                str(e),
                e.response.text,
            )
            raise
        except Exception as e:
            logging_utility.error("Unexpected error while listing tools: %s", str(e))
            raise
