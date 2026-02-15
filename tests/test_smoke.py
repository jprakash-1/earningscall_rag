"""Basic smoke test to keep the initial test pipeline green."""

from src.config import settings


def test_settings_object_exists() -> None:
    """Ensure configuration object can be imported without crashing."""

    assert settings is not None
