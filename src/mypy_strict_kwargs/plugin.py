"""
``mypy`` plugin to enforce strict keyword arguments.
"""

from collections.abc import Callable

from mypy.nodes import ArgKind
from mypy.plugin import FunctionSigContext, MethodSigContext, Plugin
from mypy.types import CallableType


def _transform_signature(
    ctx: FunctionSigContext | MethodSigContext,
) -> CallableType:
    """
    Transform positional arguments to keyword-only arguments.
    """
    original_sig: CallableType = ctx.default_signature
    new_arg_kinds: list[ArgKind] = []

    star_arg_indices = [
        index
        for index, kind in enumerate(iterable=original_sig.arg_kinds)
        if kind == ArgKind.ARG_STAR
    ]

    first_star_arg_index = star_arg_indices[0] if star_arg_indices else None

    for index, (kind, name) in enumerate(
        iterable=zip(
            original_sig.arg_kinds,
            original_sig.arg_names,
            strict=True,
        )
    ):
        # If name is None, it is a positional-only argument; leave it as is
        if name is None:
            new_arg_kinds.append(kind)

        # Transform positional arguments that can also be keyword arguments
        elif kind == ArgKind.ARG_POS:
            if first_star_arg_index is None or index > first_star_arg_index:
                new_arg_kinds.append(ArgKind.ARG_NAMED)
            else:
                new_arg_kinds.append(kind)
        elif kind == ArgKind.ARG_OPT:
            if first_star_arg_index is None or index > first_star_arg_index:
                new_arg_kinds.append(ArgKind.ARG_NAMED_OPT)
            else:
                new_arg_kinds.append(kind)
        else:
            new_arg_kinds.append(kind)

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
        return _transform_signature

    def get_method_signature_hook(
        self,
        fullname: str,
    ) -> Callable[[MethodSigContext], CallableType] | None:
        """
        Transform positional arguments to keyword-only arguments.
        """
        del self  # to satisfy vulture
        del fullname  # to satisfy vulture
        return _transform_signature


def plugin(version: str) -> type[KeywordOnlyPlugin]:
    """
    Plugin entry point.
    """
    del version  # to satisfy vulture
    return KeywordOnlyPlugin
