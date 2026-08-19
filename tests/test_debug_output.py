"""Tests for the debug output which reveals ``ignore_names`` values."""

from pathlib import Path

import pytest
from mypy import api

_SOURCE = """\
def implementation(self: object, value: int) -> None:
    return None


class Base:
    assigned = implementation

    def method(self, value: int) -> None:
        return None


class Child(Base):
    def call(self) -> None:
        super().method(1)
        super().assigned(1)


def function(value: int) -> None:
    return None


function(value=1)
"""

_CONFIG = """\
[mypy]
plugins = mypy_strict_kwargs

[mypy_strict_kwargs]
debug = true
"""


def _debug_names(
    *,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> list[str]:
    """Run ``mypy`` with debug output and return the names it wrote."""
    source_path = tmp_path / "example.py"
    source_path.write_text(data=_SOURCE, encoding="utf-8")
    config_path = tmp_path / "mypy.ini"
    config_path.write_text(data=_CONFIG, encoding="utf-8")
    api.run(
        args=[
            "--no-incremental",
            "--cache-dir",
            str(object=tmp_path / ".mypy_cache"),
            "--config-file",
            str(object=config_path),
            str(object=source_path),
        ]
    )
    prefix = "DEBUG: mypy_strict_kwargs: "
    return [
        line.removeprefix(prefix)
        for line in capsys.readouterr().err.splitlines()
        if line.startswith(prefix)
    ]


def test_super_method_fullname_is_written(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The name checked by a ``super()`` call is written."""
    names = _debug_names(tmp_path=tmp_path, capsys=capsys)

    assert "example.Base.method" in names


def test_non_method_member_fullname_is_written(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The name of a ``super()`` member which is not a method is
    written.
    """
    names = _debug_names(tmp_path=tmp_path, capsys=capsys)

    assert "example.Base.assigned" in names


def test_called_function_fullname_is_written(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The name of a called function is written."""
    names = _debug_names(tmp_path=tmp_path, capsys=capsys)

    assert "example.function" in names


def test_typeshed_names_are_not_written(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Names found while checking stubs are left out."""
    names = _debug_names(tmp_path=tmp_path, capsys=capsys)

    assert not [name for name in names if name.startswith("sys.")]
    assert not [name for name in names if name.startswith("warnings.")]
