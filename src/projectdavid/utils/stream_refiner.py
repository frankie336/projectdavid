import re
from typing import Optional

from projectdavid_common.utilities.logging_service import LoggingUtility

LOG = LoggingUtility()


class StreamRefiner:
    """
    Unifies Peeking and Suppression into a single high-speed state machine.
    Zero-latency for normal text; minimal latency only during tag detection.
    """

    TAG_OPEN = "<fc>"
    TAG_CLOSE = "</fc>"

    def __init__(self):
        self.suppressing = False
        self.hold_buf = ""  # Buffer for potential tag matches

    def process_chunk(self, text: str) -> str:
        """
        Processes a raw text chunk and returns 'clean' text for the UI.
        """
        if not text:
            return ""

        output = []

        # We iterate through the chunk to handle the case where a tag
        # is split across two chunks (e.g. chunk1: "hello <f", chunk2: "c> world")
        for char in text:
            if self.suppressing:
                # -------------------------------------------------------
                # STATE: SUPPRESSING (Waiting for </fc>)
                # -------------------------------------------------------
                self.hold_buf += char
                if self.TAG_CLOSE in self.hold_buf.lower():
                    # Found the end! Flush anything AFTER the tag and reset
                    _, after_tag = re.split(
                        re.escape(self.TAG_CLOSE), self.hold_buf, flags=re.I, maxsplit=1
                    )
                    LOG.debug("[Refiner] Tool call block closed. Resuming stream.")
                    self.hold_buf = ""
                    self.suppressing = False
                    # Note: We don't add 'after_tag' to output yet, we loop it back
                    # in case there's another tag immediately after.
                    for c in after_tag:
                        output.append(self._handle_normal_char(c))
                continue
            else:
                # -------------------------------------------------------
                # STATE: NORMAL (Streaming immediately)
                # -------------------------------------------------------
                result = self._handle_normal_char(char)
                if result:
                    output.append(result)

        return "".join(output)

    def _handle_normal_char(self, char: str) -> str:
        """Internal helper to manage the 'Potential Tag' buffer."""
        self.hold_buf += char

        # 1. Does the buffer still match the start of <fc>?
        if self.TAG_OPEN.startswith(self.hold_buf.lower()):
            if self.hold_buf.lower() == self.TAG_OPEN:
                LOG.debug("[Refiner] Tool call block detected. Suppressing...")
                self.suppressing = True
                self.hold_buf = ""  # Clear tag from stream
            return ""  # Keep holding, we're building a tag

        # 2. It's not a tag. Flush the buffer.
        flushed = self.hold_buf
        self.hold_buf = ""
        return flushed

    def flush_remaining(self) -> str:
        """Called at the very end of the stream to get any 'stuck' text."""
        if not self.suppressing and self.hold_buf:
            temp = self.hold_buf
            self.hold_buf = ""
            return temp
        return ""
