import re

from projectdavid_common.utilities.logging_service import LoggingUtility

from .function_call_suppressor import FunctionCallSuppressor

LOG = LoggingUtility()


class PeekGate:
    """
    Turns the suppressor on only if a <fc> block appears in the first
    `peek_limit` characters; otherwise passes text through unchanged.
    """

    def __init__(self, downstream: FunctionCallSuppressor, peek_limit: int = 2048):
        self.downstream = downstream
        self.peek_limit = peek_limit
        self.buf = ""
        self.mode = "peeking"  # -> "normal" after decision
        self.suppressing = False

    def feed(self, txt: str) -> str:
        # decision already taken
        if self.mode == "normal":
            return self.downstream.filter_chunk(txt) if self.suppressing else txt

        # still peeking …
        self.buf += txt
        m = re.search(r"<\s*fc\s*>", self.buf, flags=re.I)
        if m:  # found a tag
            head = self.buf[: m.start()]
            LOG.debug("[PEEK] <fc> located after leading text – engaging suppressor")
            self.suppressing = True
            self.mode = "normal"
            tail = self.buf[m.start() :]
            self.buf = ""
            return head + self.downstream.filter_chunk(tail)

        if len(self.buf) >= self.peek_limit:  # give up
            LOG.debug(
                "[PEEK] no <fc> tag within first %d chars – no suppression",
                self.peek_limit,
            )
            self.mode = "normal"
            self.suppressing = False
            out, self.buf = self.buf, ""
            return out
        return ""
