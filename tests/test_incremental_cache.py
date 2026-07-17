"""Tests for incremental cache invalidation."""

from pathlib import Path

from mypy import api


def test_plugin_configuration_invalidates_cache(tmp_path: Path) -> None:
    """Changing plugin options rechecks otherwise fresh modules."""
    source_path = tmp_path / "example.py"
    source_path.write_text(
        data="def function(value: int) -> None: ...\n\nfunction(1)\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "mypy.ini"
    config_path.write_text(
        data=(
            "[mypy]\n"
            "plugins = mypy_strict_kwargs\n"
            "strict = true\n\n"
            "[mypy_strict_kwargs]\n"
            "ignore_names = example.function\n"
        ),
        encoding="utf-8",
    )
    arguments = [
        "--cache-dir",
        str(object=tmp_path / ".mypy_cache"),
        "--config-file",
        str(object=config_path),
        str(object=source_path),
    ]

    first_stdout, first_stderr, first_status = api.run(args=arguments)

    assert first_status == 0
    assert first_stderr == ""
    assert "Success: no issues found" in first_stdout

    config_path.write_text(
        data=("[mypy]\nplugins = mypy_strict_kwargs\nstrict = true\n"),
        encoding="utf-8",
    )

    second_stdout, second_stderr, second_status = api.run(args=arguments)

    assert second_status == 1
    assert second_stderr == ""
    assert 'Too many positional arguments for "function"' in second_stdout
