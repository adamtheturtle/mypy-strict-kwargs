"""
``mypy`` plugin to enforce strict keyword arguments.
"""

from collections.abc import Callable

from mypy.nodes import ArgKind
from mypy.plugin import FunctionSigContext, Plugin
from mypy.types import CallableType


def _transform_function_signature(
    ctx: FunctionSigContext,
) -> CallableType:
    """
    Transform positional arguments to keyword-only arguments.
    """
    original_sig: CallableType = ctx.default_signature
    new_arg_kinds: list[ArgKind] = []

    for kind, name in zip(
        original_sig.arg_kinds,
        original_sig.arg_names,
        strict=True,
    ):
        # If name is None, it is a positional-only argument; leave it as is
        if name is None:
            new_arg_kinds.append(kind)

        # Transform positional arguments that can also be keyword arguments
        elif kind == ArgKind.ARG_POS:
            new_arg_kinds.append(ArgKind.ARG_NAMED)
        else:
            new_arg_kinds.append(ArgKind.ARG_NAMED_OPT)

    return original_sig.copy_modified(arg_kinds=new_arg_kinds)


class KeywordOnlyPlugin(Plugin):
    """
    A plugin that transforms positional arguments to keyword-only arguments.
    """

    def get_function_signature_hook(
        self,
        fullname: str,
    ) -> Callable[[FunctionSigContext], CallableType] | None:
        """
        Transform positional arguments to keyword-only arguments.
        """
        del self  # to satisfy vulture
        del fullname  # to satisfy vulture
        return _transform_function_signature


def plugin(version: str) -> type[KeywordOnlyPlugin]:
    """
    Plugin entry point.
    """
    del version  # to satisfy vulture
    return KeywordOnlyPlugin
