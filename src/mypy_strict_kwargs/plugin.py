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
    AssignmentStmt,
    Block,
    CallExpr,
    Expression,
    MypyFile,
    NewTypeExpr,
    ParamSpecExpr,
    RefExpr,
    Statement,
    TypeVarExpr,
    TypeVarTupleExpr,
)
from mypy.options import Options
from mypy.plugin import FunctionSigContext, MethodSigContext, Plugin
from mypy.types import CallableType

# ``TypeVar``, ``ParamSpec``, ``TypeVarTuple`` and ``NewType`` are processed
# by ``mypy``'s semantic analyzer as dedicated special-form expressions
# rather than as ordinary calls, so no call-site signature hook ever fires
# for them.  They are instead found by walking the analyzed tree.
#
# Each entry maps the analyzed expression type to the name used in error
# messages and the number of leading positional-or-keyword parameters that
# the strict-kwargs rule requires to be passed by keyword.  For the
# type-variable-like forms only ``name`` is positional-or-keyword (any
# further positional arguments are ``*constraints``, which are genuinely
# variadic).  ``NewType`` has two: ``name`` and ``tp``.
_SPECIAL_FORMS: dict[type[Expression], tuple[str, int]] = {
    TypeVarExpr: ("TypeVar", 1),
    ParamSpecExpr: ("ParamSpec", 1),
    TypeVarTupleExpr: ("TypeVarTuple", 1),
    NewTypeExpr: ("NewType", 2),
}

# Statement attributes that hold nested statements.  ``mypy``'s visitor
# classes are compiled traits that an interpreted plugin cannot subclass,
# so the tree is walked by hand following these container attributes.
_CHILD_STATEMENT_ATTRS = (
    "body",
    "else_body",
    "finally_body",
    "defs",
    "func",
    "impl",
    "items",
    "handlers",
    "bodies",
)


def _iter_statements(statements: list[Statement]) -> list[Statement]:
    """
    Return every statement reachable from ``statements``, including
    those inside function bodies, class bodies and compound statements.
    """
    collected: list[Statement] = []
    stack = list(statements)
    while stack:
        statement = stack.pop()
        collected.append(statement)
        for attr in _CHILD_STATEMENT_ATTRS:
            value: object = getattr(statement, attr, None)
            children = (
                cast("list[object]", value)
                if isinstance(value, list)
                else [value]
            )
            for child in children:
                if isinstance(child, Block):
                    stack.extend(child.body)
                elif isinstance(child, Statement):
                    stack.append(child)
    return collected


def _find_special_form_violations(
    tree: MypyFile,
    *,
    ignore_names: list[str],
) -> list[tuple[CallExpr, str]]:
    """
    Return ``(call, display_name)`` pairs for special-form definitions
    in ``tree`` that pass positional arguments where the strict-kwargs
    rule requires keyword arguments.  ``ignore_names`` holds fully
    qualified names to skip.
    """
    violations: list[tuple[CallExpr, str]] = []
    for statement in _iter_statements(statements=tree.defs):
        if not isinstance(statement, AssignmentStmt):
            continue
        call = statement.rvalue
        if not isinstance(call, CallExpr) or call.analyzed is None:
            continue
        for expr_type, (display_name, prefix) in _SPECIAL_FORMS.items():
            if not isinstance(call.analyzed, expr_type):
                continue
            callee = call.callee
            if isinstance(callee, RefExpr) and callee.fullname in ignore_names:
                break
            uses_positional = any(
                call.arg_kinds[index] == ArgKind.ARG_POS
                for index in range(min(prefix, len(call.arg_kinds)))
            )
            if uses_positional:
                violations.append((call, display_name))
            break
    return violations


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

    original_sig: CallableType = ctx.default_signature
    new_arg_kinds: list[ArgKind] = []

    star_arg_indices = [
        index
        for index, kind in enumerate(iterable=original_sig.arg_kinds)
        if kind == ArgKind.ARG_STAR
    ]

    first_star_arg_index = star_arg_indices[0] if star_arg_indices else None

    # Some methods get called with a positional argument that we do not supply.
    skip_first_argument_suffixes = (
        # Gets called when an instance of the class is called.
        ".__call__",
        # Descriptor attribute access
        ".__get__",
        # Descriptor attribute assignment
        ".__set__",
    )
    skip_first_argument = fullname.endswith(skip_first_argument_suffixes)

    skip_second_argument_suffixes = (
        # Descriptor attribute access.
        # The second argument is the instance of the class.
        ".__get__",
        # Descriptor attribute assignment.
        # The second argument is the value to be assigned.
        ".__set__",
    )

    skip_second_argument = fullname.endswith(skip_second_argument_suffixes)

    for index, (kind, name) in enumerate(
        iterable=zip(
            original_sig.arg_kinds,
            original_sig.arg_names,
            strict=True,
        )
    ):
        if skip_first_argument and index == 0:
            new_arg_kinds.append(kind)
            continue

        if skip_second_argument and index == 1:
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
    return original_sig.copy_modified(
        arg_kinds=new_arg_kinds,  # pyrefly: ignore[bad-argument-type]
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
        self._special_form_violations: dict[
            str, list[tuple[CallExpr, str]]
        ] = {}

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

    def _check_special_forms(
        self,
        ctx: FunctionSigContext | MethodSigContext,
    ) -> None:
        """
        Flag special-form definitions that pass positional arguments.

        ``TypeVar``, ``ParamSpec``, ``TypeVarTuple`` and ``NewType`` are
        processed by ``mypy``'s semantic analyzer and never reach a
        call-site signature hook, so the module being type checked is
        walked to report any that pass positional arguments where
        keyword arguments are required.

        Signature hooks may run inside speculative, error-suppressed
        type checking (for example, during operator or overload
        resolution), so the cached violations are re-reported on every
        invocation.  ``mypy`` drops the suppressed copies and removes
        the remaining duplicates.

        A module that contains no normally type checked call never
        triggers a signature hook, so its special forms are not walked.
        See https://github.com/adamtheturtle/mypy-strict-kwargs/issues/454.
        """
        # ``mypy`` always calls ``set_modules`` before any signature hook
        # fires, so ``_modules`` is populated by the time this runs.
        # ``pylint`` cannot introspect the compiled ``mypy.plugin.Plugin``
        # base class, so it does not know ``_modules`` is a mapping.
        all_modules = cast("dict[str, MypyFile]", self._modules)
        modules = all_modules.values()  # pylint: disable=no-member

        path = ctx.api.path
        if path not in self._special_form_violations:
            tree: MypyFile | None = next(
                (module for module in modules if module.path == path),
                None,
            )
            if tree is not None and not tree.is_stub:
                self._special_form_violations[path] = (
                    _find_special_form_violations(
                        tree=tree,
                        ignore_names=self._ignore_names,
                    )
                )
            else:
                self._special_form_violations[path] = []

        for call, display_name in self._special_form_violations[path]:
            if self._debug:
                sys.stderr.write(
                    f"DEBUG: mypy_strict_kwargs: {display_name} "
                    f"at {path}:{call.line}\n"
                )
            ctx.api.fail(
                f'Too many positional arguments for "{display_name}"',
                call,
                code=MISC,
            )

    def _signature_hook(
        self,
        ctx: FunctionSigContext | MethodSigContext,
        *,
        fullname: str,
    ) -> CallableType:
        """Check special forms, then transform the call signature."""
        self._check_special_forms(ctx=ctx)
        return _transform_signature(
            ctx=ctx,
            fullname=fullname,
            ignore_names=self._ignore_names,
            debug=self._debug,
        )

    def get_function_signature_hook(
        self,
        fullname: str,
    ) -> Callable[[FunctionSigContext], CallableType] | None:
        """Transform positional arguments to keyword-only arguments."""
        return partial(self._signature_hook, fullname=fullname)

    def get_method_signature_hook(
        self,
        fullname: str,
    ) -> Callable[[MethodSigContext], CallableType] | None:
        """Transform positional arguments to keyword-only arguments."""
        return partial(self._signature_hook, fullname=fullname)


def plugin(version: str) -> type[KeywordOnlyPlugin]:
    """Plugin entry point."""
    del version  # to satisfy vulture
    return KeywordOnlyPlugin
