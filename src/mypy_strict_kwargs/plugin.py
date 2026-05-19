"""``mypy`` plugin to enforce strict keyword arguments."""

import configparser
import sys
import tomllib
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

from mypy.nodes import ArgKind, Decorator, FuncDef, OverloadedFuncDef, TypeInfo
from mypy.options import Options
from mypy.plugin import (
    ClassDefContext,
    FunctionSigContext,
    MethodSigContext,
    Plugin,
)
from mypy.types import (
    CallableType,
    FunctionLike,
    Overloaded,
    ProperType,
    Type,
    get_proper_type,
)


def _preserved_positional_argument_count(fullname: str) -> int:
    """Return positional arguments to keep positional after method binding."""
    # Some methods get called with positional arguments that callers do not
    # supply explicitly.
    if fullname.endswith((".__get__", ".__set__")):
        # Descriptor attribute access and assignment.
        return 2
    if fullname.endswith(".__call__"):
        # Called implicitly when an instance of the class is called.
        return 1
    return 0


def _transform_signature(
    ctx: FunctionSigContext | MethodSigContext,
    fullname: str,
    *,
    ignore_names: list[str],
    debug: bool,
) -> CallableType:
    """Transform positional arguments to keyword-only arguments."""
    if debug:
        sys.stderr.write(f"DEBUG: mypy_strict_kwargs: {fullname}\n")

    return _transform_callable_type(
        signature=ctx.default_signature,
        fullname=fullname,
        ignore_names=ignore_names,
        skip_bound_argument=False,
    )


def _transform_callable_type(
    *,
    signature: CallableType,
    fullname: str,
    ignore_names: list[str],
    skip_bound_argument: bool,
) -> CallableType:
    """Transform positional arguments in a callable type."""
    new_arg_kinds: list[ArgKind] = []

    star_arg_indices = [
        index
        for index, kind in enumerate(iterable=signature.arg_kinds)
        if kind == ArgKind.ARG_STAR
    ]

    first_star_arg_index = star_arg_indices[0] if star_arg_indices else None

    preserved_positional_argument_count = _preserved_positional_argument_count(
        fullname=fullname,
    )
    skip_offset = 1 if skip_bound_argument else 0
    skip_indices = {
        index + skip_offset
        for index in range(preserved_positional_argument_count)
    }

    if skip_bound_argument:
        skip_indices.add(0)

    for index, (kind, name) in enumerate(
        iterable=zip(
            signature.arg_kinds,
            signature.arg_names,
            strict=True,
        )
    ):
        if index in skip_indices:
            new_arg_kinds.append(kind)
            continue

        # If name is None, it is a positional-only argument; leave it as is
        is_positional_only = name is None
        should_ignore = fullname in ignore_names
        if is_positional_only or should_ignore:
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

    # See https://github.com/facebook/pyrefly/issues/1995.
    return signature.copy_modified(
        arg_kinds=new_arg_kinds,  # pyrefly: ignore[bad-argument-type]
    )


def _transform_function_like(
    *,
    signature: FunctionLike,
    fullname: str,
    ignore_names: list[str],
    skip_bound_argument: bool,
) -> FunctionLike:
    """Transform positional arguments in any function-like type."""
    if isinstance(signature, CallableType):
        return _transform_callable_type(
            signature=signature,
            fullname=fullname,
            ignore_names=ignore_names,
            skip_bound_argument=skip_bound_argument,
        )

    overloaded = cast("Overloaded", signature)
    return Overloaded(
        items=[
            _transform_callable_type(
                signature=item,
                fullname=fullname,
                ignore_names=ignore_names,
                skip_bound_argument=skip_bound_argument,
            )
            for item in overloaded.items
        ],
    )


def _transform_type(
    *,
    typ: Type | None,
    fullname: str,
    ignore_names: list[str],
    skip_bound_argument: bool,
) -> ProperType | None:
    """Transform positional arguments in a method type."""
    if typ is None:
        return None

    proper_type = get_proper_type(typ=typ)
    return _transform_function_like(
        signature=cast("FunctionLike", proper_type),
        fullname=fullname,
        ignore_names=ignore_names,
        skip_bound_argument=skip_bound_argument,
    )


def _transform_base_method_signatures(
    ctx: ClassDefContext,
    *,
    ignore_names: list[str],
) -> None:
    """Transform method definitions used by ``super()`` member access."""
    for info in ctx.cls.info.mro[1:]:
        _transform_type_info_methods(type_info=info, ignore_names=ignore_names)


def _transform_type_info_methods(
    *,
    type_info: TypeInfo,
    ignore_names: list[str],
) -> None:
    """Transform methods on a type info in place."""
    for symbol in type_info.names.values():
        node = symbol.node

        if isinstance(node, (FuncDef, OverloadedFuncDef)):
            node.type = _transform_type(
                typ=node.type,
                fullname=node.fullname,
                ignore_names=ignore_names,
                skip_bound_argument=node.has_self_or_cls_argument,
            )
        elif isinstance(node, Decorator):
            node.func.type = _transform_type(
                typ=node.func.type,
                fullname=node.fullname,
                ignore_names=ignore_names,
                skip_bound_argument=node.func.has_self_or_cls_argument,
            )
            node.var.type = _transform_type(
                typ=node.var.type,
                fullname=node.fullname,
                ignore_names=ignore_names,
                skip_bound_argument=node.func.has_self_or_cls_argument,
            )


class KeywordOnlyPlugin(Plugin):
    """
    A plugin that transforms positional arguments to keyword-only
    arguments.
    """

    def __init__(self, options: Options) -> None:
        """Configure the plugin.

        This is not friendly to errors yet.
        """
        super().__init__(options=options)
        self._ignore_names: list[str] = []
        self._debug = False

        if options.config_file is None:
            return

        config_file = Path(options.config_file)
        if config_file.suffix == ".toml":
            with config_file.open(mode="rb") as rf:
                config_dictionary = tomllib.load(rf)

            tools = dict(config_dictionary.get("tool", {}))
            plugin_config = dict(tools.get("mypy_strict_kwargs", {}))
            self._ignore_names = list(plugin_config.get("ignore_names", []))
            self._debug = bool(plugin_config.get("debug", False))
        else:
            # Handle ``mypy.ini``, ``.mypy.ini``, ``setup.cfg``
            parser = configparser.ConfigParser()
            parser.read(filenames=config_file)

            if parser.has_section(section="mypy_strict_kwargs"):
                ignore_names_str = parser.get(
                    section="mypy_strict_kwargs",
                    option="ignore_names",
                    fallback="",
                )
                if ignore_names_str:
                    self._ignore_names = [
                        name.strip()
                        for name in ignore_names_str.split(sep=",")
                        if name.strip()
                    ]
                self._debug = bool(
                    parser.getboolean(
                        section="mypy_strict_kwargs",
                        option="debug",
                        fallback=False,
                    )
                )

    def get_function_signature_hook(
        self,
        fullname: str,
    ) -> Callable[[FunctionSigContext], CallableType] | None:
        """Transform positional arguments to keyword-only arguments."""
        return partial(
            _transform_signature,
            fullname=fullname,
            ignore_names=self._ignore_names,
            debug=self._debug,
        )

    def get_method_signature_hook(
        self,
        fullname: str,
    ) -> Callable[[MethodSigContext], CallableType] | None:
        """Transform positional arguments to keyword-only arguments."""
        return partial(
            _transform_signature,
            fullname=fullname,
            ignore_names=self._ignore_names,
            debug=self._debug,
        )

    def get_base_class_hook(
        self,
        fullname: str,
    ) -> Callable[[ClassDefContext], None] | None:
        """Transform base methods used by ``super()`` member access."""
        del fullname
        return partial(
            _transform_base_method_signatures,
            ignore_names=self._ignore_names,
        )


def plugin(version: str) -> type[KeywordOnlyPlugin]:
    """Plugin entry point."""
    del version  # to satisfy vulture
    return KeywordOnlyPlugin
