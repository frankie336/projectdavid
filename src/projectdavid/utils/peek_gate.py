# ------------------------------------------------------------------
# utils.peek_gate  (progressive flush version)
# ------------------------------------------------------------------
import re

from projectdavid_common.utilities.logging_service import LoggingUtility

from .function_call_suppressor import FunctionCallSuppressor

LOG = LoggingUtility()


class PeekGate:
    """
    • Streams *most* text immediately while the gate decides whether
      a <fc> block will appear.
    • Keeps only SAFETY_MARGIN bytes back so a tag split across two
      chunks can still be caught.
    """

    SAFETY_MARGIN = 8  # enough to hold "<fc>" + whitespace

    def __init__(self, downstream: FunctionCallSuppressor, peek_limit: int = 2048):
        self.downstream = downstream
        self.peek_limit = peek_limit
        self.buf = ""
        self.mode = "peeking"  # → "normal" after decision
        self.suppressing = False

    # ----------------------------------------------------------
    def _safe_flush(self, new_txt: str = "") -> str:
        """
        Return everything except the last SAFETY_MARGIN chars,
        which are kept for tag-boundary safety.
        """
        self.buf += new_txt
        if len(self.buf) <= self.SAFETY_MARGIN:
            return ""
        flush_len = len(self.buf) - self.SAFETY_MARGIN
        head, self.buf = self.buf[:flush_len], self.buf[flush_len:]
        return head

    # ----------------------------------------------------------
    def feed(self, txt: str) -> str:
        # decision already taken
        if self.mode == "normal":
            return self.downstream.filter_chunk(txt) if self.suppressing else txt

        # still peeking …
        self.buf += txt
        m = re.search(r"<\s*fc\s*>", self.buf, flags=re.I)
        if m:  # tag found
            head = self.buf[: m.start()]
            LOG.debug("[PEEK] <fc> located – engaging suppressor")
            self.suppressing = True
            self.mode = "normal"
            tail, self.buf = self.buf[m.start() :], ""
            return head + self.downstream.filter_chunk(tail)

        # no tag yet – overflow?
        if len(self.buf) >= self.peek_limit:
            LOG.debug(
                "[PEEK] no <fc> tag within first %d chars – " "streaming normally",
                self.peek_limit,
            )
            flush, self.mode = self.buf, "normal"
            self.buf = ""
            return flush

        # still undecided → flush safe part
        return self._safe_flush()
