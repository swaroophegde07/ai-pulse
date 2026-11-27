import pytest


@pytest.fixture(autouse=True)
def _no_real_groq_key(monkeypatch):
    """Keep a developer's real GROQ_API_KEY from enabling network calls in tests."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
