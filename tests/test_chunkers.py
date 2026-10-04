from rag_lab.ingest.chunkers import RecursiveChunker
from rag_lab.schemas import Chunk, Document


def make_doc(text: str, doc_id: str = "doc1") -> Document:
    return Document(doc_id=doc_id, source_path=f"{doc_id}.txt", raw_text=text)


def test_empty_document_returns_no_chunks():
    chunker = RecursiveChunker()
    assert chunker.chunk(make_doc("")) == []
    assert chunker.chunk(make_doc("   \n\n  ")) == []


def test_short_document_returns_single_chunk():
    chunker = RecursiveChunker(target_tokens=400, min_tokens=5)
    chunks = chunker.chunk(make_doc("This is a short document."))
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "This is a short document."
    assert chunks[0].position == 0


def test_long_document_returns_multiple_chunks():
    para = " ".join(["word"] * 50)
    text = "\n\n".join([para] * 4)
    chunker = RecursiveChunker(target_tokens=60, overlap_tokens=0, min_tokens=10)
    chunks = chunker.chunk(make_doc(text))
    assert len(chunks) >= 2
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_chunks_respect_target_size():
    para = " ".join(["word"] * 50)
    text = "\n\n".join([para] * 6)
    chunker = RecursiveChunker(target_tokens=100, overlap_tokens=0, min_tokens=10)
    chunks = chunker.chunk(make_doc(text))
    for c in chunks:
        assert c.token_count <= 120, f"chunk too big: {c.token_count}"


def test_overlap_between_adjacent_chunks():
    para = " ".join([f"w{i}" for i in range(50)])
    text = "\n\n".join([para] * 3)
    chunker = RecursiveChunker(target_tokens=60, overlap_tokens=10, min_tokens=5)
    chunks = chunker.chunk(make_doc(text))
    assert len(chunks) >= 2
    tail = chunks[0].text.split()[-5:]
    head = chunks[1].text.split()[:15]
    assert any(w in head for w in tail), "expected overlap between chunks"


def test_chunk_ids_are_deterministic():
    text = "\n\n".join(["word " * 50] * 3)
    doc = make_doc(text)
    chunker = RecursiveChunker(target_tokens=80, overlap_tokens=0)
    first = chunker.chunk(doc)
    second = chunker.chunk(doc)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_metadata_propagates_from_document():
    doc = Document(
        doc_id="doc42",
        source_path="paper.pdf",
        title="A Paper",
        raw_text="Some content here.",
        metadata={"author": "Ada"},
    )
    chunker = RecursiveChunker(min_tokens=1)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata.get("author") == "Ada"
    assert chunks[0].metadata.get("source_path") == "paper.pdf"
    assert chunks[0].metadata.get("title") == "A Paper"


def test_chunk_ids_match_helper():
    doc = make_doc("A paragraph. " * 30)
    chunker = RecursiveChunker(target_tokens=50, overlap_tokens=0)
    chunks = chunker.chunk(doc)
    for i, c in enumerate(chunks):
        assert c.chunk_id == Chunk.make_chunk_id(doc.doc_id, i)