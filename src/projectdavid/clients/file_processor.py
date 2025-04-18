import asyncio
import csv
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

try:
    from typing import LiteralString  # Python 3.11+
except ImportError:
    from typing_extensions import LiteralString

import numpy as np
import pdfplumber
import validators
from projectdavid_common import UtilsInterface
from sentence_transformers import SentenceTransformer

log = UtilsInterface.LoggingUtility()


class FileProcessor:
    def __init__(self, max_workers: int = 4, chunk_size: int = 512):
        self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.embedding_model_name = "paraphrase-MiniLM-L6-v2"
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # compute token limits
        self.max_seq_length = self.embedding_model.get_max_seq_length()
        self.special_tokens_count = 2
        self.effective_max_length = self.max_seq_length - self.special_tokens_count

        # chunk_size cannot exceed 4× model max
        self.chunk_size = min(chunk_size, self.effective_max_length * 4)

        log.info("Initialized optimized FileProcessor")

    def validate_file(self, file_path: Path):
        """Ensure file exists and is under 100 MB."""
        max_size = 100 * 1024 * 1024
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.stat().st_size > max_size:
            mb = max_size // (1024 * 1024)
            raise ValueError(f"{file_path.name} > {mb} MB limit")

    def _detect_file_type(self, file_path: Path) -> str:
        """Return 'pdf', 'text', or 'csv'."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        if suffix == ".csv":
            return "csv"
        if suffix in {".txt", ".md", ".rst"}:
            return "text"
        return "unknown"

    async def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Async entrypoint: validate, detect type, then dispatch to the
        appropriate processor (_process_pdf, _process_text, or _process_csv).
        """
        file_path = Path(file_path)
        self.validate_file(file_path)
        ftype = self._detect_file_type(file_path)

        if ftype == "pdf":
            return await self._process_pdf(file_path)
        if ftype == "text":
            return await self._process_text(file_path)
        if ftype == "csv":
            return await self._process_csv(file_path)
        raise ValueError(f"Unsupported extension: {file_path.suffix}")

    # ——— PDF / TEXT pipelines unchanged ——— #

    async def _process_pdf(self, file_path: Path) -> Dict[str, Any]:
        page_chunks, doc_meta = await self._extract_text(file_path)
        all_chunks, line_data = [], []

        for page_text, page_num, line_nums in page_chunks:
            lines = page_text.split("\n")
            buf, buf_lines, length = [], [], 0

            for line, ln in zip(lines, line_nums):
                l = len(line) + 1
                if length + l <= self.chunk_size:
                    buf.append(line)
                    buf_lines.append(ln)
                    length += l
                else:
                    if buf:
                        all_chunks.append("\n".join(buf))
                        line_data.append({"page": page_num, "lines": buf_lines})
                        buf, buf_lines, length = [], [], 0

                    # split any oversized line
                    for piece in self._split_oversized_chunk(line):
                        all_chunks.append(piece)
                        line_data.append({"page": page_num, "lines": [ln]})

            if buf:
                all_chunks.append("\n".join(buf))
                line_data.append({"page": page_num, "lines": buf_lines})

        vectors = await asyncio.gather(
            *[self._encode_chunk_async(c) for c in all_chunks]
        )

        return {
            "content": "\n\n".join(all_chunks),
            "metadata": {
                **doc_meta,
                "source": str(file_path),
                "chunks": len(all_chunks),
                "type": "pdf",
            },
            "chunks": all_chunks,
            "vectors": [v.tolist() for v in vectors],
            "line_data": line_data,
        }

    async def _process_text(self, file_path: Path) -> Dict[str, Any]:
        text, extra_meta, _ = await self._extract_text(file_path)
        chunks = self._chunk_text(text)
        vectors = await asyncio.gather(*[self._encode_chunk_async(c) for c in chunks])
        return {
            "content": text,
            "metadata": {
                **extra_meta,
                "source": str(file_path),
                "chunks": len(chunks),
                "type": "text",
            },
            "chunks": chunks,
            "vectors": [v.tolist() for v in vectors],
        }

    # ——— NEW: CSV pipeline ——— #

    async def _process_csv(
        self, file_path: Path, text_field: str = "description"
    ) -> Dict[str, Any]:
        """
        Read each row, embed the `text_field`, and collect per-row metadata
        from all other columns.
        """
        # load rows synchronously
        rows, texts, metas = [], [], []
        with file_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                txt = row.get(text_field, "").strip()
                if not txt:
                    continue
                texts.append(txt)
                # all other columns become metadata
                row_meta = {k: v for k, v in row.items() if k != text_field and v}
                metas.append(row_meta)

        # embed in parallel
        vectors = await asyncio.gather(*[self._encode_chunk_async(t) for t in texts])

        return {
            "content": None,  # CSVs may not have monolithic text
            "metadata": {
                "source": str(file_path),
                "rows": len(texts),
                "type": "csv",
            },
            "chunks": texts,
            "vectors": [v.tolist() for v in vectors],
            "csv_row_metadata": metas,
        }

    # ——— shared helpers ——— #

    async def _extract_text(self, file_path: Path) -> Union[
        Tuple[List[Tuple[str, int, List[int]]], Dict[str, Any]],
        Tuple[str, Dict[str, Any], List[int]],
    ]:
        loop = asyncio.get_event_loop()
        if file_path.suffix.lower() == ".pdf":
            return await loop.run_in_executor(
                self._executor, self._extract_pdf_text, file_path
            )
        else:
            text = await loop.run_in_executor(
                self._executor, self._read_text_file, file_path
            )
            return text, {}, []

    def _extract_pdf_text(self, file_path: Path):
        page_chunks, meta = [], {}
        with pdfplumber.open(file_path) as pdf:
            meta.update(
                {
                    "author": pdf.metadata.get("Author", ""),
                    "title": pdf.metadata.get("Title", file_path.stem),
                    "page_count": len(pdf.pages),
                }
            )
            for i, page in enumerate(pdf.pages, start=1):
                lines = page.extract_text_lines()
                txts, nums = [], []
                # sort by vertical position
                sorted_lines = sorted(lines, key=lambda x: x["top"])
                # enumerate to get a reliable line number
                for ln_idx, L in enumerate(sorted_lines, start=1):
                    t = L.get("text", "").strip()
                    if t:
                        txts.append(t)
                        nums.append(ln_idx)
                if txts:
                    page_chunks.append(("\n".join(txts), i, nums))
        return page_chunks, meta

    def _read_text_file(self, file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="latin-1")

    async def _encode_chunk_async(self, chunk: str) -> np.ndarray:
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self.embedding_model.encode(
                [chunk],
                convert_to_numpy=True,
                truncate="model_max_length",
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0],
        )

    def _chunk_text(self, text: str) -> List[str]:
        # split into sentences, then re-chunk to token limits
        sentences = re.split(r"(?<=[\.!?])\s+", text)
        chunks, buf, length = [], [], 0

        for sent in sentences:
            slen = len(sent) + 1
            if length + slen <= self.chunk_size:
                buf.append(sent)
                length += slen
            else:
                if buf:
                    chunks.append(" ".join(buf))
                    buf, length = [], 0
                # sentence itself may be too big
                while len(sent) > self.chunk_size:
                    part, sent = sent[: self.chunk_size], sent[self.chunk_size :]
                    chunks.append(part)
                buf, length = [sent], len(sent)

        if buf:
            chunks.append(" ".join(buf))

        return chunks

    def _split_oversized_chunk(self, chunk: str, tokens: List[str] = None) -> List[str]:
        if tokens is None:
            tokens = self.embedding_model.tokenizer.tokenize(chunk)
        out = []
        for i in range(0, len(tokens), self.effective_max_length):
            seg = tokens[i : i + self.effective_max_length]
            out.append(self.embedding_model.tokenizer.convert_tokens_to_string(seg))
        return out
