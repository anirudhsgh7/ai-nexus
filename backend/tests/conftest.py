import json
from pathlib import Path

import pytest

from app.config import Settings, clear_settings_cache
from app.llm.ollama import OllamaProvider

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def load_fixture_json(name: str) -> dict:
    return json.loads(load_fixture(name))


@pytest.fixture(autouse=True)
def _clean_settings_cache():
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
async def provider(settings: Settings):
    p = OllamaProvider(settings)
    yield p
    await p.aclose()


@pytest.fixture
def chat_response() -> dict:
    return load_fixture_json("chat_response.json")
