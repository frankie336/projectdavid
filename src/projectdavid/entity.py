import os
from typing import Optional

from dotenv import load_dotenv
from projectdavid_common import UtilsInterface

# Use relative imports for modules within your package.
from .clients.actions import ActionsClient
from .clients.assistants import AssistantsClient
from .clients.files import FileClient
from .clients.inference import InferenceClient
from .clients.messages import MessagesClient
from .clients.runs import RunsClient
from .clients.synchronous_inference_stream import SynchronousInferenceStream
from .clients.threads import ThreadsClient
from .clients.tools import ToolsClient
from .clients.users import UsersClient
from .clients.vectors import VectorStoreClient
from .utils.run_monitor import HttpRunMonitor

# Load environment variables from .env file.
load_dotenv()

# Initialize logging utility.
logging_utility = UtilsInterface.LoggingUtility()


class Entity:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the main client with configuration.
        Optionally, a configuration object can be injected to decouple from environment variables.
        """
        self.base_url = base_url or os.getenv(
            "ASSISTANTS_BASE_URL", "http://localhost:9000/"
        )
        self.api_key = api_key or os.getenv("API_KEY", "your_api_key")

        logging_utility.info("Entity initialized with base_url: %s", self.base_url)

        # Lazy initialization caches for service instances.
        self._users_client: Optional[UsersClient] = None
        self._assistants_client: Optional[AssistantsClient] = None
        self._tool_service: Optional[ToolsClient] = None
        self._thread_service: Optional[ThreadsClient] = None
        self._messages_client: Optional[MessagesClient] = None

        self._runs_client: Optional[RunsClient] = None
        self._actions_client: Optional[ActionsClient] = None
        self._inference_client: Optional[InferenceClient] = None
        self._file_client: Optional[FileClient] = None
        self._vectors_client: Optional[VectorStoreClient] = None

        self._synchronous_inference_stream: Optional[SynchronousInferenceStream] = None

        # Utils
        self._run_monitor: Optional[HttpRunMonitor] = None

    @property
    def users(self) -> UsersClient:
        if self._users_client is None:
            self._users_client = UsersClient(
                base_url=self.base_url, api_key=self.api_key
            )
        return self._users_client

    @property
    def assistants(self) -> AssistantsClient:
        if self._assistants_client is None:
            self._assistants_client = AssistantsClient(
                base_url=self.base_url, api_key=self.api_key
            )
        return self._assistants_client

    @property
    def tools(self) -> ToolsClient:
        if self._tool_service is None:
            self._tool_service = ToolsClient(
                base_url=self.base_url, api_key=self.api_key
            )
        return self._tool_service

    @property
    def threads(self) -> ThreadsClient:
        if self._thread_service is None:
            self._thread_service = ThreadsClient(
                base_url=self.base_url, api_key=self.api_key
            )
        return self._thread_service

    @property
    def messages(self) -> MessagesClient:
        if self._messages_client is None:
            self._messages_client = MessagesClient(
                base_url=self.base_url, api_key=self.api_key
            )
        return self._messages_client

    def submit_function_call_output(self, thread, assistant_id, tool_id, content):

        self._messages_client.submit_tool_output(thread, assistant_id, tool_id, content)

    @property
    def runs(self) -> RunsClient:
        if self._runs_client is None:
            self._runs_client = RunsClient(base_url=self.base_url, api_key=self.api_key)
        return self._runs_client

    @property
    def actions(self) -> ActionsClient:
        if self._actions_client is None:
            self._actions_client = ActionsClient(
                base_url=self.base_url, api_key=self.api_key
            )
        return self._actions_client

    @property
    def inference(self) -> InferenceClient:
        if self._inference_client is None:
            self._inference_client = InferenceClient(
                base_url=self.base_url, api_key=self.api_key
            )
        return self._inference_client

    @property
    def synchronous_inference_stream(self) -> SynchronousInferenceStream:
        if self._synchronous_inference_stream is None:
            self._synchronous_inference_stream = SynchronousInferenceStream(
                self.inference
            )
        return self._synchronous_inference_stream

    @property
    def files(self) -> FileClient:
        if self._file_client is None:
            self._file_client = FileClient(base_url=self.base_url, api_key=self.api_key)
        return self._file_client

    @property
    def vectors(self) -> VectorStoreClient:
        if self._vectors_client is None:
            self._vectors_client = VectorStoreClient(
                base_url=self.base_url, api_key=self.api_key
            )

        return self._vectors_client
