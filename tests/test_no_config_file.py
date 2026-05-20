"""Test plugin behavior when no configuration file is provided."""

import importlib

from mypy.options import Options
from mypy.types import AnyType, TypeOfAny

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


def test_transform_type_leaves_non_function_types_unchanged() -> None:
    """Test that non-function proper types are left unchanged."""
    typ = AnyType(type_of_any=TypeOfAny.explicit)
    plugin_module = importlib.import_module(name="mypy_strict_kwargs.plugin")
    transform_type = vars(plugin_module)["_transform_type"]

    transformed_type = transform_type(
        typ=typ,
        fullname="test.value",
        ignore_names=[],
        skip_bound_argument=False,
    )

    assert transformed_type is typ
