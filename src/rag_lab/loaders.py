"""File loaders. Each loader converts a file on disk into a Document."""

from pathlib import Path

from rag_lab.schemas import Document

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def _extract_title(text: str, fallback: str) -> str:
    """Return the first markdown heading, first non-empty line, or the fallback."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
        return stripped[:120]
    return fallback


def load_file(path: Path, raw_root: Path) -> Document | None:
    """Load one file into a Document.

    Returns None if the extension is unsupported or the file is empty.
    `raw_root` is used to compute a stable, relative source path.
    """
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try latin-1 as a fallback for files that aren't clean UTF-8.
        text = path.read_text(encoding="latin-1")

    if not text.strip():
        return None

    rel_path = path.relative_to(raw_root).as_posix()
    doc_id = Document.make_doc_id(rel_path)
    title = _extract_title(text, fallback=path.stem)

    return Document(
        doc_id=doc_id,
        source_path=rel_path,
        title=title,
        raw_text=text,
        metadata={"extension": path.suffix.lower(), "size_bytes": path.stat().st_size},
    )