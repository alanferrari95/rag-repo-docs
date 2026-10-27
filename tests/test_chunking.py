from ragdocs.chunking import chunk_corpus, chunk_document
from ragdocs.corpus import DEFAULT_CORPUS, Document, load_corpus


def test_sections_keep_their_heading_path(docs):
    chunks = chunk_document(docs[0])
    assert [c.heading for c in chunks] == ["Runbook", "Runbook > Canary", "Runbook > Rollback"]
    assert [c.chunk_id for c in chunks] == [f"demo/runbook.md#{i}" for i in range(3)]


def test_hash_inside_code_fence_is_not_a_heading(docs):
    rollback = chunk_document(docs[0])[-1]
    assert "# this is a comment, not a heading" in rollback.text


def test_embed_text_carries_repo_path_and_heading(docs):
    chunk = chunk_document(docs[0])[1]
    assert chunk.embed_text.startswith("demo > runbook.md > Runbook > Canary\n\n")


def test_long_section_is_split_with_overlap():
    paragraphs = [" ".join(f"p{i}w{j}" for j in range(30)) for i in range(10)]  # 300 words
    doc = Document("demo/long.md", "demo", "long.md", "# Long\n\n" + "\n\n".join(paragraphs))
    chunks = chunk_document(doc, max_words=100, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 100 + 40 for c in chunks)
    # each chunk after the first starts with the last paragraph of the one before it
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.text.split("\n\n")[0] == prev.text.split("\n\n")[-1]


def test_oversized_block_becomes_overlapping_windows():
    words = [f"w{i}" for i in range(500)]
    doc = Document("demo/wall.md", "demo", "wall.md", "# Wall\n\n" + " ".join(words))
    chunks = chunk_document(doc, max_words=200, overlap=50)
    text = " ".join(c.text for c in chunks).split()
    assert set(text) == set(words)  # nothing lost
    assert len(text) > len(words)  # windows overlap


def test_real_corpus_loads_and_chunks():
    docs = load_corpus(DEFAULT_CORPUS)
    assert len(docs) == 16
    chunks = chunk_corpus(docs)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert max(len(c.text.split()) for c in chunks) <= 220 + 40
