# src/projectdavid/utils/network_device_handler.py
import json
import logging
import re

# Ensure consumers have netmiko installed (pip install netmiko)
try:
    from netmiko import (
        ConnectHandler,
        NetmikoAuthenticationException,
        NetmikoTimeoutException,
    )
except ImportError:
    ConnectHandler = None

log = logging.getLogger(__name__)


class NetworkDeviceHandler:
    """
    Curated SDK tool handler for executing network commands securely.
    Implements the "Store & Slice" architecture locally, ensuring massive
    CLI outputs do not bloat the LLM context window.
    """

    def __init__(self, credential_provider_callback):
        """
        :param credential_provider_callback: A function that takes a `hostname` (str)
               and returns a Netmiko connection dictionary (device_type, host, username, password, etc.)
        """
        if ConnectHandler is None:
            raise ImportError(
                "The 'netmiko' library is required to use the NetworkDeviceHandler. "
                "Please run: pip install netmiko"
            )
        self.credential_provider = credential_provider_callback

    def __call__(self, tool_name: str, arguments: dict) -> str:
        """
        Matches the signature expected by `event.execute(handler)`.
        """
        if tool_name != "run_network_commands":
            return json.dumps(
                {"status": "error", "error": f"Unsupported tool: {tool_name}"}
            )

        hostname = arguments.get("hostname")
        commands = arguments.get("commands", [])
        filter_pattern = arguments.get("filter_pattern")

        if not hostname or not commands:
            return json.dumps(
                {
                    "status": "error",
                    "error": "Missing 'hostname' or 'commands' argument.",
                }
            )

        log.info(
            f"[NetworkDeviceHandler] Connecting to {hostname} to run {len(commands)} commands..."
        )

        # 1. Fetch consumer's local credentials
        try:
            device_params = self.credential_provider(hostname)
            if not device_params:
                raise ValueError(f"No credentials found for {hostname}")

            # Ensure host is set if the consumer only provided auth details
            if "host" not in device_params:
                device_params["host"] = hostname

        except Exception as e:
            return json.dumps(
                {"status": "error", "error": f"Credential lookup failed: {str(e)}"}
            )

        # 2. Execute & Slice
        results = {}
        try:
            with ConnectHandler(**device_params) as net_connect:
                for cmd in commands:
                    log.debug(f"[NetworkDeviceHandler] Running: {cmd}")
                    raw_output = net_connect.send_command(cmd)

                    # --- THE "STORE & SLICE" FILTERING LOGIC ---
                    if filter_pattern:
                        try:
                            regex = re.compile(filter_pattern, re.IGNORECASE)
                            sliced_lines = [
                                line
                                for line in raw_output.splitlines()
                                if regex.search(line)
                            ]

                            if not sliced_lines:
                                results[cmd] = (
                                    f"No lines matched filter: '{filter_pattern}'"
                                )
                            else:
                                results[cmd] = "\n".join(sliced_lines)
                        except re.error:
                            results[cmd] = (
                                f"⚠️ Invalid Regex Filter provided by AI: '{filter_pattern}'. Raw output suppressed to save context."
                            )
                    else:
                        results[cmd] = raw_output

            return json.dumps(
                {"status": "success", "hostname": hostname, "data": results}
            )

        except NetmikoAuthenticationException:
            return json.dumps(
                {
                    "status": "error",
                    "error": "Authentication failed. Bad username/password.",
                }
            )
        except NetmikoTimeoutException:
            return json.dumps(
                {
                    "status": "error",
                    "error": "Connection timed out. Device unreachable.",
                }
            )
        except Exception as e:
            return json.dumps(
                {"status": "error", "error": f"Execution failed: {str(e)}"}
            )
