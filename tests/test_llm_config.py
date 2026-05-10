"""Tests for provider configuration and OpenAI-compatible LLM wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from linux_agent.config import load_config
from linux_agent.graph import build_graph


def test_load_config_reads_deepseek_env_vars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("workspace_root: .\n", encoding="utf-8")

    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")

    config = load_config(str(config_file))

    assert config.workspace_root == tmp_path.resolve()
    assert config.llm_base_url == "https://api.deepseek.com/v1"
    assert config.llm_api_key == "deepseek-test-key"


def test_load_config_yaml_overrides_env(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "workspace_root: .",
                "llm_model: deepseek-chat",
                "llm_base_url: https://custom.example/v1",
                "llm_api_key: yaml-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")

    config = load_config(str(config_file))

    assert config.llm_model == "deepseek-chat"
    assert config.llm_base_url == "https://custom.example/v1"
    assert config.llm_api_key == "yaml-key"


def test_build_graph_passes_provider_config_to_chatopenai(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "workspace_root: .",
                "llm_model: deepseek-chat",
                "llm_base_url: https://api.deepseek.com/v1",
                "llm_api_key: deepseek-test-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_config(str(config_file))

    with patch("linux_agent.graph.ChatOpenAI") as chat_openai:
        build_graph(config)

    chat_openai.assert_called_once_with(
        model="deepseek-chat",
        temperature=0.0,
        base_url="https://api.deepseek.com/v1",
        api_key="deepseek-test-key",
    )