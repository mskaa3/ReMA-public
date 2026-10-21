"""Keep mock validation tests offline; IDS integration tests opt in explicitly."""
import pytest


@pytest.fixture(autouse=True)
def offline_ids_default(monkeypatch):
    monkeypatch.setenv("REMA_IDS_ENABLED", "false")
