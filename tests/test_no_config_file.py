"""Test plugin behavior when no configuration file is provided."""

import importlib
from typing import Any

import pytest
from mypy.options import Options

from mypy_strict_kwargs.plugin import KeywordOnlyPlugin


def test_no_config_file() -> None:
    """
    Test that plugin provides hooks when no configuration file
    exists.
    """
    options = Options()

    plugin = KeywordOnlyPlugin(options=options)

    assert plugin.get_base_class_hook(fullname="test.Base") is not None
    assert plugin.get_function_signature_hook(fullname="test.func") is not None
    assert plugin.get_method_signature_hook(fullname="test.method") is not None


def test_unsupported_pattern_raises() -> None:
    """Test that unsupported pattern nodes fail explicitly."""
    unsupported_pattern: Any = object()
    plugin_module = importlib.import_module(name="mypy_strict_kwargs.plugin")
    collect_call_exprs_from_pattern = plugin_module.__dict__[
        "_collect_call_exprs_from_pattern"
    ]

    with pytest.raises(
        expected_exception=TypeError,
        match="Unsupported match pattern: object",
    ):
        collect_call_exprs_from_pattern(unsupported_pattern, [])
