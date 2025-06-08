import re

from projectdavid_common.utilities.logging_service import LoggingUtility

LOG = LoggingUtility()


# ─────────────────────────────────────────────────────────────────────
#  function-call filter helpers  (unchanged except for logger)
# ─────────────────────────────────────────────────────────────────────
class FunctionCallSuppressor:
    OPEN_RE = re.compile(r"<\s*fc\s*>", re.I)
    CLOSE_RE = re.compile(r"</\s*fc\s*>", re.I)

    def __init__(self):
        self.in_fc = False
        self.buf = ""

    def filter_chunk(self, chunk: str) -> str:
        self.buf += chunk
        out = ""

        while self.buf:
            if not self.in_fc:
                m = self.OPEN_RE.search(self.buf)
                if not m:
                    out += self.buf
                    self.buf = ""
                    break
                out += self.buf[: m.start()]
                LOG.debug("[SUPPRESSOR] <fc> detected")
                self.buf = self.buf[m.end() :]
                self.in_fc = True
            else:
                m = self.CLOSE_RE.search(self.buf)
                if not m:
                    break  # wait for more tokens
                LOG.debug("[SUPPRESSOR] </fc> detected — block suppressed")
                self.buf = self.buf[m.end() :]
                self.in_fc = False
        return out
