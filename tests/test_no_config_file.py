"""Test plugin behavior when no configuration file is provided."""

from mypy.options import Options

from mypy_strict_kwargs.plugin import KeywordOnlyPlugin


def test_no_config_file() -> None:
    """
    Test that plugin provides hooks when no configuration file
    exists.
    """
    options = Options()

    plugin = KeywordOnlyPlugin(options=options)

    assert plugin.get_function_signature_hook(fullname="test.func") is not None
    assert plugin.get_method_signature_hook(fullname="test.method") is not None
