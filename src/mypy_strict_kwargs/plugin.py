"""``mypy`` plugin to enforce strict keyword arguments."""

import configparser
import sys
import tomllib
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

from mypy.errorcodes import MISC
from mypy.nodes import (
    ArgKind,
    CallExpr,
    Decorator,
    FuncDef,
    MemberExpr,
    NameExpr,
    Node,
    OverloadedFuncDef,
    SuperExpr,
)
from mypy.options import Options
from mypy.plugin import (
    ClassDefContext,
    FunctionSigContext,
    MethodSigContext,
    Plugin,
)
from mypy.types import CallableType, FunctionLike, Overloaded

_AST_CHILD_ATTRS = frozenset(
    {
        "analyzed",
        "args",
        "base",
        "body",
        "callee",
        "condition",
        "defs",
        "else_body",
        "expr",
        "expressions",
        "if_expr",
        "index",
        "indices",
        "items",
        "left",
        "lvalues",
        "op",
        "operands",
        "right",
        "rvalue",
        "value",
        "values",
    },
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


def _callable_items(signature: FunctionLike) -> list[CallableType]:
    """Return the callable items in a function-like signature."""
    if isinstance(signature, CallableType):
        return [signature]
    if isinstance(signature, Overloaded):
        return list(signature.items)

    raise TypeError(type(signature).__name__)


def _is_super_expr(expr: SuperExpr | CallExpr) -> bool:
    """Return whether an expression is ``super`` or ``super()``."""
    if isinstance(expr, SuperExpr):
        return True

    callee = expr.callee
    return isinstance(callee, NameExpr) and (
        callee.fullname == "builtins.super" or callee.name == "super"
    )


def _super_method_name(expr: CallExpr) -> str | None:
    """Return the method name for a ``super().method(...)`` call."""
    if isinstance(expr.callee, SuperExpr):
        return expr.callee.name
    if not isinstance(expr.callee, MemberExpr):
        return None
    if not isinstance(expr.callee.expr, (CallExpr, SuperExpr)):
        return None
    if not _is_super_expr(expr=expr.callee.expr):
        return None
    return expr.callee.name


def _call_disallows_positional_argument(
    *,
    call: CallExpr,
    signature: CallableType,
    fullname: str,
    ignore_names: list[str],
    skip_bound_argument: bool,
) -> bool:
    """Return whether a call passes a transformed argument by position."""
    transformed = _transform_callable_type(
        signature=signature,
        fullname=fullname,
        ignore_names=ignore_names,
        skip_bound_argument=skip_bound_argument,
    )

    positional_argument_index = 0
    for actual_arg_kind in call.arg_kinds:
        if actual_arg_kind != ArgKind.ARG_POS:
            continue

        formal_arg_index = positional_argument_index
        if skip_bound_argument:
            formal_arg_index += 1
        positional_argument_index += 1

        if formal_arg_index >= len(transformed.arg_kinds):
            return False

        if transformed.arg_kinds[formal_arg_index] in {
            ArgKind.ARG_NAMED,
            ArgKind.ARG_NAMED_OPT,
        }:
            return True

    return False


def _super_call_disallows_positional_argument(
    *,
    call: CallExpr,
    signature: FunctionLike,
    fullname: str,
    ignore_names: list[str],
    skip_bound_argument: bool,
) -> bool:
    """Return whether every overload rejects positional super
    arguments.
    """
    return all(
        _call_disallows_positional_argument(
            call=call,
            signature=callable_item,
            fullname=fullname,
            ignore_names=ignore_names,
            skip_bound_argument=skip_bound_argument,
        )
        for callable_item in _callable_items(signature=signature)
    )


def _iter_child_nodes(node: Node) -> list[Node]:
    """Return syntax-tree child nodes relevant to call-expression
    traversal.
    """
    children: list[Node] = []
    for attr_name in _AST_CHILD_ATTRS:
        if not hasattr(node, attr_name):
            continue

        value: object = getattr(node, attr_name)
        if isinstance(value, Node):
            children.append(value)
        elif isinstance(value, (list, tuple)):
            children.extend(
                item
                for item in cast("list[object] | tuple[object, ...]", value)
                if isinstance(item, Node)
            )

    return children


def _iter_call_exprs(node: Node) -> list[CallExpr]:
    """Return call expressions contained in a node."""
    calls: list[CallExpr] = []
    seen: set[int] = set()
    stack = [node]

    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        if isinstance(current, CallExpr):
            calls.append(current)

        stack.extend(_iter_child_nodes(node=current))

    return calls


def _check_super_method_call(
    *,
    ctx: ClassDefContext,
    expr: CallExpr,
    method_name: str,
    ignore_names: list[str],
) -> None:
    """Check one ``super()`` method call expression."""
    for info in ctx.cls.info.mro[1:]:
        symbol = info.names.get(method_name)
        if symbol is None:
            continue

        node = symbol.node
        if isinstance(node, (FuncDef, OverloadedFuncDef)):
            fullname = node.fullname
            typ = node.type
            skip_bound_argument = node.has_self_or_cls_argument
        elif isinstance(node, Decorator):
            fullname = node.fullname
            typ = node.func.type
            skip_bound_argument = node.func.has_self_or_cls_argument
        else:
            continue

        if fullname in ignore_names:
            return

        if not isinstance(typ, FunctionLike):
            return

        if _super_call_disallows_positional_argument(
            call=expr,
            signature=typ,
            fullname=fullname,
            ignore_names=ignore_names,
            skip_bound_argument=skip_bound_argument,
        ):
            ctx.api.fail(
                msg=(
                    f'Too many positional arguments for "{method_name}" '
                    f'of "{info.name}"'
                ),
                ctx=expr,
                code=MISC,
            )
            return


def _check_super_method_calls(
    ctx: ClassDefContext,
    *,
    ignore_names: list[str],
) -> None:
    """Check ``super()`` method calls in a class body."""
    for expr in _iter_call_exprs(node=ctx.cls.defs):
        method_name = _super_method_name(expr=expr)
        if method_name is None:
            continue
        _check_super_method_call(
            ctx=ctx,
            expr=expr,
            method_name=method_name,
            ignore_names=ignore_names,
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
        """Check ``super()`` method calls without mutating base
        classes.
        """
        del fullname
        return partial(
            _check_super_method_calls,
            ignore_names=self._ignore_names,
        )


def plugin(version: str) -> type[KeywordOnlyPlugin]:
    """Plugin entry point."""
    del version  # to satisfy vulture
    return KeywordOnlyPlugin
