# tests/e2e/test_projectdavid_flow.py
import os
import sys
import time

from dotenv import load_dotenv

from projectdavid import (  # Assuming your library is installable as 'projectdavid'
    Entity,
)

# Load .env file if it exists (useful for local testing)
load_dotenv()

# --- Configuration ---
# Get configuration from environment variables, falling back to defaults if needed
BASE_URL = os.getenv("PROJECTDAVID_BASE_URL", "http://localhost:9000")
ENTITIES_API_KEY = os.getenv("ENTITIES_API_KEY")
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")
# Use a default or environment variable for user_id
DEFAULT_USER_ID = (
    "user_kUKV8octgG2aMc7kxAcD3i"  # Consider if this should be dynamic or from env
)
USER_ID = os.getenv("PROJECTDAVID_USER_ID", DEFAULT_USER_ID)
ASSISTANT_NAME = "ci_test_assistant_" + str(int(time.time()))  # Unique name for CI
MODEL_PROVIDER = os.getenv("PROJECTDAVID_MODEL_PROVIDER", "Hyperbolic")
MODEL_NAME = os.getenv("PROJECTDAVID_MODEL_NAME", "hyperbolic/Qwen/QwQ-32B-Preview")


# --- Helper Function for Validation ---
def check_env_vars():
    """Check if required environment variables are set."""
    required_vars = ["ENTITIES_API_KEY", "HYPERBOLIC_API_KEY"]
    missing_vars = [var for var in required_vars if not globals().get(var)]
    if missing_vars:
        print(
            f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}"
        )
        print("Please set them in your environment or a .env file.")
        sys.exit(1)  # Exit with error code
    print("✅ Required environment variables found.")


def run_flow_test():
    """Executes the end-to-end test flow."""
    print(f"--- Starting ProjectDavid Flow Test ---")
    print(f"Using Base URL: {BASE_URL}")
    print(f"Using User ID: {USER_ID}")

    # Initialize client
    try:
        client = Entity(base_url=BASE_URL, api_key=ENTITIES_API_KEY)
        print("✅ Client initialized.")
    except Exception as e:
        print(f"❌ Error initializing client: {e}")
        sys.exit(1)

    # -----------------------------
    # Create an assistant
    # ------------------------------
    try:
        print(f"Creating assistant '{ASSISTANT_NAME}'...")
        assistant = client.assistants.create_assistant(
            name=ASSISTANT_NAME,
            instructions="You are a helpful AI assistant",
        )
        print(f"✅ Created assistant with ID: {assistant.id}")
    except Exception as e:
        print(f"❌ Error creating assistant: {e}")
        # Optionally try to clean up if needed, though failure here is critical
        sys.exit(1)

    # -----------------------------------------------
    # Create a thread
    # ------------------------------------------------
    actual_thread_id = None  # Initialize
    try:
        print("Creating thread...")
        thread = client.threads.create_thread(participant_ids=[USER_ID])
        actual_thread_id = thread.id  # Store the dynamically created thread ID
        print(f"✅ Created thread with ID: {actual_thread_id}")
    except Exception as e:
        print(f"❌ Error creating thread: {e}")
        # Clean up assistant if thread creation fails?
        # client.assistants.delete_assistant(assistant.id) ?
        sys.exit(1)

    # -----------------------------------------
    # Create a message using the NEW thread ID
    # --------------------------------------------
    message = None  # Initialize
    try:
        print(f"Creating message in thread {actual_thread_id}...")
        message = client.messages.create_message(
            thread_id=actual_thread_id,
            role="user",
            content="Explain a black hole to me in pure mathematical terms",
            assistant_id=assistant.id,
        )
        print(f"✅ Created message with ID: {message.id}")
    except Exception as e:
        print(f"❌ Error creating message: {e}")
        # Clean up assistant and thread?
        sys.exit(1)

    # ---------------------------------------------
    # Create a run using the NEW thread ID
    # ----------------------------------------------
    run = None  # Initialize
    try:
        print(f"Creating run in thread {actual_thread_id}...")
        run = client.runs.create_run(
            assistant_id=assistant.id, thread_id=actual_thread_id
        )
        print(f"✅ Created run with ID: {run.id}")
    except Exception as e:
        print(f"❌ Error creating run: {e}")
        # Clean up assistant, thread, message?
        sys.exit(1)

    # ------------------------------------------------
    # Instantiate the synchronous streaming helper
    # --------------------------------------------------
    sync_stream = client.synchronous_inference_stream
    print("✅ Synchronous stream helper instantiated.")

    # ------------------------------------------------------
    # Set up the stream using the NEW thread ID
    # --------------------------------------------------------
    try:
        print(f"Setting up stream for thread {actual_thread_id}...")
        sync_stream.setup(
            user_id=USER_ID,
            thread_id=actual_thread_id,
            assistant_id=assistant.id,
            message_id=message.id,
            run_id=run.id,
            api_key=HYPERBOLIC_API_KEY,  # Pass the specific key here
        )
        print("✅ Stream setup complete. Starting streaming...")
    except Exception as e:
        print(f"❌ Error setting up stream: {e}")
        # Clean up?
        sys.exit(1)

    # --- Stream initial LLM response ---
    stream_successful = False
    output_buffer = ""
    try:
        for chunk in sync_stream.stream_chunks(
            provider=MODEL_PROVIDER,
            model=MODEL_NAME,
            timeout_per_chunk=20.0,  # Increased timeout slightly
        ):
            content = chunk.get("content", "")
            if content:
                print(content, end="", flush=True)
                output_buffer += content
                stream_successful = True  # Mark as successful if we receive any content
        print("\n--- End of Stream ---")
        if not stream_successful:
            print("⚠️ Warning: Stream completed but received no content.")
            # Decide if this is an error condition for your use case
            # sys.exit(1)
        else:
            print("✅ Stream completed successfully with content.")

    except Exception as e:
        print(f"\n❌ --- Stream Error: {e} ---")
        # Clean up?
        sys.exit(1)  # Exit with error on stream failure

    # --- Optional Cleanup ---
    # If you want tests to clean up resources (good practice but adds complexity)
    # try:
    #     print(f"\nAttempting cleanup for assistant {assistant.id} and thread {actual_thread_id}...")
    #     # Delete run? (API might not support this)
    #     # Delete message? (API might not support this)
    #     client.threads.delete_thread(actual_thread_id)
    #     print(f"✅ Deleted thread {actual_thread_id}")
    #     client.assistants.delete_assistant(assistant.id)
    #     print(f"✅ Deleted assistant {assistant.id}")
    # except Exception as e:
    #     print(f"⚠️ Warning: Cleanup failed: {e}")

    print("--- ProjectDavid Flow Test Completed Successfully ---")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting test script execution...")
    check_env_vars()
    run_flow_test()
    print("Script finished.")
    sys.exit(0)  # Explicitly exit with success code
