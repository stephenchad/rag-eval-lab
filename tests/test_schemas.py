from rag_lab.schemas import Chunk, Document


def test_doc_id_is_deterministic():
    a = Document.make_doc_id("data/raw/paper1.pdf")
    b = Document.make_doc_id("data/raw/paper1.pdf")
    assert a == b


def test_doc_id_differs_for_different_paths():
    a = Document.make_doc_id("data/raw/paper1.pdf")
    b = Document.make_doc_id("data/raw/paper2.pdf")
    assert a != b


def test_chunk_id_is_deterministic():
    a = Chunk.make_chunk_id("doc123", 5)
    b = Chunk.make_chunk_id("doc123", 5)
    assert a == b


def test_chunk_id_changes_with_position():
    a = Chunk.make_chunk_id("doc123", 5)
    b = Chunk.make_chunk_id("doc123", 6)
    assert a != b


def test_document_roundtrip():
    doc = Document(
        doc_id=Document.make_doc_id("x.txt"),
        source_path="x.txt",
        raw_text="hello world",
    )
    dumped = doc.model_dump_json()
    restored = Document.model_validate_json(dumped)
    assert restored.doc_id == doc.doc_id
    assert restored.raw_text == "hello world"


def test_chunk_roundtrip():
    chunk = Chunk(
        chunk_id=Chunk.make_chunk_id("doc1", 0),
        doc_id="doc1",
        text="some text",
        position=0,
        token_count=2,
    )
    dumped = chunk.model_dump_json()
    restored = Chunk.model_validate_json(dumped)
    assert restored.chunk_id == chunk.chunk_id
    assert restored.text == "some text"