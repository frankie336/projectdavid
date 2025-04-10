import os

from dotenv import load_dotenv

load_dotenv()

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}
ACTION_REQUIRED_STATUS = "pending_action"

PLATFORM_TOOLS = ["code_interpreter", "web_search", "vector_store_search", "computer"]


TOOLS_ID_MAP = {
    "code_interpreter": "tool_79YkQEz5cDwpJjnR7oJ80D",
    "web_search": "tool_BiIwycpLo1n5Dh6BHN01v8",
    "vector_store_search": "tool_MCaJpXJU3eW6vaMUybEf6i",
    "computer": "tool_PJQ6VcnkmRCMankObjtRcn",
}


SPECIAL_CASE_TOOL_HANDLING = ["computer", "code_interpreter"]

ERROR_NO_CONTENT = (
    "ERROR: The Tool has failed to return any content. The current stage of the workflow is tool submission. "
    "Please inform the user."
)


DIRECT_DATABASE_URL = (
    "mysql+pymysql://ollama:3e4Qv5uo2Cg31zC1@localhost:3307/cosmic_catalyst"
)


# ------------------------------------------------
# Vendors sometimes have clashing model names.
# This can interfere with routing logic
# _________________________________________________
MODEL_MAP = {
    "deepseek-ai/deepseek-reasoner": "deepseek-reasoner",
    "deepseek-ai/deepseek-chat": "deepseek-chat",
    # Deepseek@TogetherAi
    "together-ai/deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "together-ai/deepseek-ai/DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    # Llama@TogetherAi
    "together-ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "together-ai/meta-llama/Llama-4-Scout-17B-16E-Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # Deepseek@Hyperbolic
    "hyperbolic/deepseek-ai/deepseek-ai/DeepSeek-V3-0324": "deepseek-ai/DeepSeek-V3-0324",
    "hyperbolic/deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "hyperbolic/deepseek-ai/DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    # Llama@Hyperbolic
    "hyperbolic/meta-llama/Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "hyperbolic/meta-llama/Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "hyperbolic/meta-llama/Meta-Llama-3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "hyperbolic/meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "hyperbolic/meta-llama/Meta-Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "hyperbolic/meta-llama/Meta-Llama-3-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    # Quen@Hyperbolic
    # --- Add Quen models here if needed ---
    # --- Google Gemini & Gemma Models ---
    "google/gemini-1.0-pro-vision-latest": "gemini-1.0-pro-vision-latest",
    "google/gemini-pro-vision": "gemini-pro-vision",
    "google/gemini-1.5-pro-latest": "gemini-1.5-pro-latest",
    "google/gemini-1.5-pro-001": "gemini-1.5-pro-001",
    "google/gemini-1.5-pro-002": "gemini-1.5-pro-002",
    "google/gemini-1.5-pro": "gemini-1.5-pro",
    "google/gemini-1.5-flash-latest": "gemini-1.5-flash-latest",
    "google/gemini-1.5-flash-001": "gemini-1.5-flash-001",
    "google/gemini-1.5-flash-001-tuning": "gemini-1.5-flash-001-tuning",
    "google/gemini-1.5-flash": "gemini-1.5-flash",
    "google/gemini-1.5-flash-002": "gemini-1.5-flash-002",
    "google/gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
    "google/gemini-1.5-flash-8b-001": "gemini-1.5-flash-8b-001",
    "google/gemini-1.5-flash-8b-latest": "gemini-1.5-flash-8b-latest",
    "google/gemini-1.5-flash-8b-exp-0827": "gemini-1.5-flash-8b-exp-0827",
    "google/gemini-1.5-flash-8b-exp-0924": "gemini-1.5-flash-8b-exp-0924",
    "google/gemini-2.5-pro-exp-03-25": "gemini-2.5-pro-exp-03-25",
    "google/gemini-2.5-pro-preview-03-25": "gemini-2.5-pro-preview-03-25",
    "google/gemini-2.0-flash-exp": "gemini-2.0-flash-exp",
    "google/gemini-2.0-flash": "gemini-2.0-flash",
    "google/gemini-2.0-flash-001": "gemini-2.0-flash-001",
    "google/gemini-2.0-flash-exp-image-generation": "gemini-2.0-flash-exp-image-generation",
    "google/gemini-2.0-flash-lite-001": "gemini-2.0-flash-lite-001",
    "google/gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
    "google/gemini-2.0-flash-lite-preview-02-05": "gemini-2.0-flash-lite-preview-02-05",
    "google/gemini-2.0-flash-lite-preview": "gemini-2.0-flash-lite-preview",
    "google/gemini-2.0-pro-exp": "gemini-2.0-pro-exp",
    "google/gemini-2.0-pro-exp-02-05": "gemini-2.0-pro-exp-02-05",
    "google/gemini-exp-1206": "gemini-exp-1206",
    "google/gemini-2.0-flash-thinking-exp-01-21": "gemini-2.0-flash-thinking-exp-01-21",
    "google/gemini-2.0-flash-thinking-exp": "gemini-2.0-flash-thinking-exp",
    "google/gemini-2.0-flash-thinking-exp-1219": "gemini-2.0-flash-thinking-exp-1219",
    "google/learnlm-1.5-pro-experimental": "learnlm-1.5-pro-experimental",
    "google/gemma-3-1b-it": "gemma-3-1b-it",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "google/gemma-3-12b-it": "gemma-3-12b-it",
    "google/gemma-3-27b-it": "gemma-3-27b-it",
}  # End MODEL_MAP


WEB_SEARCH_BASE_URL = "http://localhost:8080/"

# Extend SUPPORTED_MIME_TYPES and define helper
SUPPORTED_MIME_TYPES = {
    ".c": "text/x-c",
    ".cpp": "text/x-c++",
    ".cs": "text/x-csharp",
    ".css": "text/css",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".go": "text/x-golang",
    ".html": "text/html",
    ".java": "text/x-java",
    ".js": "text/javascript",
    ".json": "application/json",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".php": "text/x-php",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".py": "text/x-python",
    ".pyx": "text/x-script.python",
    ".rb": "text/x-ruby",
    ".sh": "application/x-sh",
    ".tex": "text/x-tex",
    ".ts": "application/typescript",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".zip": "application/zip",
    ".tar": "application/x-tar",
    ".gz": "application/gzip",
    ".rar": "application/vnd.rar",
    ".7z": "application/x-7z-compressed",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
}


def get_mime_type(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return SUPPORTED_MIME_TYPES.get(ext.lower())


BROWSER_RENDERABLE_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".html",
    ".htm",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
}

# For text/* MIME types, define allowed encodings
ALLOWED_TEXT_ENCODINGS = ["utf-8", "utf-16", "ascii"]
