"""Tests for plugin configuration validation."""

from pathlib import Path

from mypy import api

_SOURCE = "def function(value: int) -> None: ...\n\nfunction(1)\n"

# ``mypy`` exits with this status for a fatal configuration error.
_CONFIGURATION_ERROR_STATUS = 2


def _run_mypy(*, tmp_path: Path, config: str, config_name: str) -> str:
    """Run ``mypy`` with a configuration file and return its errors."""
    source_path = tmp_path / "example.py"
    source_path.write_text(data=_SOURCE, encoding="utf-8")
    config_path = tmp_path / config_name
    config_path.write_text(data=config, encoding="utf-8")
    _, stderr, status = api.run(
        args=[
            "--cache-dir",
            str(object=tmp_path / ".mypy_cache"),
            "--config-file",
            str(object=config_path),
            str(object=source_path),
        ]
    )
    assert status == _CONFIGURATION_ERROR_STATUS
    return stderr


def test_toml_scalar_plugin_section(tmp_path: Path) -> None:
    """A scalar plugin section is reported as a configuration error."""
    stderr = _run_mypy(
        tmp_path=tmp_path,
        config=(
            "[tool.mypy]\n"
            'plugins = ["mypy_strict_kwargs"]\n\n'
            "[tool]\n"
            'mypy_strict_kwargs = "invalid"\n'
        ),
        config_name="pyproject.toml",
    )

    assert "[tool.mypy_strict_kwargs]: expected a table" in stderr


def test_toml_ignore_names_string(tmp_path: Path) -> None:
    """A string ``ignore_names`` is reported as a configuration error."""
    stderr = _run_mypy(
        tmp_path=tmp_path,
        config=(
            "[tool.mypy]\n"
            'plugins = ["mypy_strict_kwargs"]\n\n'
            "[tool.mypy_strict_kwargs]\n"
            'ignore_names = "example.function"\n'
        ),
        config_name="pyproject.toml",
    )

    assert (
        '[tool.mypy_strict_kwargs]: "ignore_names" must be an array of strings'
    ) in stderr


def test_toml_ignore_names_non_string_item(tmp_path: Path) -> None:
    """A non-string ``ignore_names`` item is a configuration error."""
    stderr = _run_mypy(
        tmp_path=tmp_path,
        config=(
            "[tool.mypy]\n"
            'plugins = ["mypy_strict_kwargs"]\n\n'
            "[tool.mypy_strict_kwargs]\n"
            "ignore_names = [1]\n"
        ),
        config_name="pyproject.toml",
    )

    assert (
        '[tool.mypy_strict_kwargs]: "ignore_names" must be an array of strings'
    ) in stderr


def test_toml_debug_string(tmp_path: Path) -> None:
    """A string ``debug`` value is reported as a configuration error."""
    stderr = _run_mypy(
        tmp_path=tmp_path,
        config=(
            "[tool.mypy]\n"
            'plugins = ["mypy_strict_kwargs"]\n\n'
            "[tool.mypy_strict_kwargs]\n"
            'debug = "false"\n'
        ),
        config_name="pyproject.toml",
    )

    assert '[tool.mypy_strict_kwargs]: "debug" must be a boolean' in stderr


def test_ini_debug_not_a_boolean(tmp_path: Path) -> None:
    """A non-boolean INI ``debug`` value is a configuration error."""
    stderr = _run_mypy(
        tmp_path=tmp_path,
        config=(
            "[mypy]\n"
            "plugins = mypy_strict_kwargs\n\n"
            "[mypy_strict_kwargs]\n"
            "debug = notabool\n"
        ),
        config_name="mypy.ini",
    )

    assert '[mypy_strict_kwargs]: "debug" must be a boolean' in stderr
