"""
Shared test fixtures.

`fake_encode` is a deterministic bag-of-hashed-words embedding used in
tests instead of the real sentence-transformers model, so retrieval
tests run fast, offline, and deterministically.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIM = 64


def fake_encode(texts):
    """Deterministic, dependency-free 'embedding': counts hashed words
    into fixed buckets and L2-normalizes. Similar text -> similar
    vector, which is all the retrieval logic under test actually needs."""
    vectors = np.zeros((len(texts), FIXTURE_DIM))
    for i, text in enumerate(texts):
        for word in text.lower().split():
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % FIXTURE_DIM
            vectors[i, bucket] += 1
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@pytest.fixture
def sample_pdf_path():
    """One real contract from data/contracts, used where a real PDF is
    useful (extraction + chunking tests)."""
    path = Path(__file__).parent.parent / "data" / "contracts" / "NON disclosure agreement Edited.pdf"
    assert path.exists(), f"Sample contract not found at {path}"
    return str(path)


@pytest.fixture(autouse=True)
def isolate_log_file(tmp_path, monkeypatch):
    """Redirect the default log path to a temp file for every test,
    so tests never depend on or pollute the real logs."""
    import src.critique as critique_module
    import src.retrieve as retrieve_module
    import src.route as route_module

    temp_log = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(retrieve_module, "DEFAULT_LOG_PATH", temp_log)
    monkeypatch.setattr(route_module, "DEFAULT_LOG_PATH", temp_log)
    monkeypatch.setattr(critique_module, "DEFAULT_LOG_PATH", temp_log)
    return temp_log
