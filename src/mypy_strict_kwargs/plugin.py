"""``mypy`` plugin to enforce strict keyword arguments."""

import configparser
import sys
import tomllib
from collections.abc import Callable
from functools import partial
from pathlib import Path

from mypy.errorcodes import MISC
from mypy.nodes import (
    REVEAL_TYPE,
    ArgKind,
    AssertStmt,
    AssertTypeExpr,
    AssignmentExpr,
    AssignmentStmt,
    AwaitExpr,
    Block,
    CallExpr,
    CastExpr,
    ClassDef,
    ComparisonExpr,
    ConditionalExpr,
    Decorator,
    DelStmt,
    DictExpr,
    DictionaryComprehension,
    Expression,
    ExpressionStmt,
    ForStmt,
    FuncDef,
    FuncItem,
    GeneratorExpr,
    IfStmt,
    IndexExpr,
    LambdaExpr,
    ListComprehension,
    ListExpr,
    MatchStmt,
    MemberExpr,
    Node,
    OperatorAssignmentStmt,
    OpExpr,
    OverloadedFuncDef,
    RaiseStmt,
    ReturnStmt,
    RevealExpr,
    SetComprehension,
    SetExpr,
    SliceExpr,
    StarExpr,
    Statement,
    SuperExpr,
    TemplateStrExpr,
    TryStmt,
    TupleExpr,
    TypeApplication,
    UnaryExpr,
    WhileStmt,
    WithStmt,
    YieldExpr,
    YieldFromExpr,
)
from mypy.options import Options
from mypy.plugin import (
    ClassDefContext,
    FunctionSigContext,
    MethodSigContext,
    Plugin,
)
from mypy.types import CallableType, FunctionLike


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


def _super_method_name(expr: CallExpr) -> str | None:
    """Return the method name for a ``super().method(...)`` call."""
    match expr.callee:
        case SuperExpr(name=name):
            return name
        case _:
            return None


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
    callable_items = (
        [signature] if isinstance(signature, CallableType) else signature.items
    )
    return all(
        _call_disallows_positional_argument(
            call=call,
            signature=callable_item,
            fullname=fullname,
            ignore_names=ignore_names,
            skip_bound_argument=skip_bound_argument,
        )
        for callable_item in callable_items
    )


def _collect_call_exprs(item: object, calls: list[CallExpr], /) -> None:
    """Collect call expressions from a syntax-tree node or expression."""
    if isinstance(item, CallExpr):
        calls.append(item)
        _collect_call_exprs(item.callee, calls)
        for argument in item.args:
            _collect_call_exprs(argument, calls)
        if item.analyzed is not None:
            _collect_call_exprs(item.analyzed, calls)
        return

    if isinstance(item, Statement):
        _collect_call_exprs_from_statement(item, calls)
        return

    if isinstance(item, Expression):
        _collect_call_exprs_from_expression(item, calls)


def _collect_call_exprs_from_statement(  # noqa: C901, PLR0912, PLR0915  # pylint: disable=too-complex,too-many-branches,too-many-statements
    statement: Statement,
    calls: list[CallExpr],
    /,
) -> None:
    """Collect call expressions from a statement."""
    if isinstance(statement, ExpressionStmt):
        _collect_call_exprs(statement.expr, calls)
    elif isinstance(statement, AssignmentStmt):
        _collect_call_exprs(statement.rvalue, calls)
        for lvalue in statement.lvalues:
            _collect_call_exprs(lvalue, calls)
    elif isinstance(statement, OperatorAssignmentStmt):
        _collect_call_exprs(statement.rvalue, calls)
        _collect_call_exprs(statement.lvalue, calls)
    elif isinstance(statement, WhileStmt):
        _collect_call_exprs(statement.expr, calls)
        _collect_call_exprs(statement.body, calls)
        if statement.else_body is not None:
            _collect_call_exprs(statement.else_body, calls)
    elif isinstance(statement, ForStmt):
        _collect_call_exprs(statement.index, calls)
        _collect_call_exprs(statement.expr, calls)
        _collect_call_exprs(statement.body, calls)
        if statement.else_body is not None:
            _collect_call_exprs(statement.else_body, calls)
    elif isinstance(statement, ReturnStmt):
        if statement.expr is not None:
            _collect_call_exprs(statement.expr, calls)
    elif isinstance(statement, AssertStmt):
        _collect_call_exprs(statement.expr, calls)
        if statement.msg is not None:
            _collect_call_exprs(statement.msg, calls)
    elif isinstance(statement, DelStmt):
        _collect_call_exprs(statement.expr, calls)
    elif isinstance(statement, IfStmt):
        for expression in statement.expr:
            _collect_call_exprs(expression, calls)
        for body in statement.body:
            _collect_call_exprs(body, calls)
        if statement.else_body is not None:
            _collect_call_exprs(statement.else_body, calls)
    elif isinstance(statement, RaiseStmt):
        if statement.expr is not None:
            _collect_call_exprs(statement.expr, calls)
        if statement.from_expr is not None:
            _collect_call_exprs(statement.from_expr, calls)
    elif isinstance(statement, TryStmt):
        _collect_call_exprs(statement.body, calls)
        for handler_type, handler in zip(
            statement.types,
            statement.handlers,
            strict=True,
        ):
            if handler_type is not None:
                _collect_call_exprs(handler_type, calls)
            _collect_call_exprs(handler, calls)
        for variable in statement.vars:
            if variable is not None:
                _collect_call_exprs(variable, calls)
        if statement.else_body is not None:
            _collect_call_exprs(statement.else_body, calls)
        if statement.finally_body is not None:
            _collect_call_exprs(statement.finally_body, calls)
    elif isinstance(statement, WithStmt):
        for expression, target in zip(
            statement.expr,
            statement.target,
            strict=True,
        ):
            _collect_call_exprs(expression, calls)
            if target is not None:
                _collect_call_exprs(target, calls)
        _collect_call_exprs(statement.body, calls)
    elif isinstance(statement, MatchStmt):
        _collect_call_exprs(statement.subject, calls)
        for pattern, guard, body in zip(
            statement.patterns,
            statement.guards,
            statement.bodies,
            strict=True,
        ):
            _collect_call_exprs(pattern, calls)
            if guard is not None:
                _collect_call_exprs(guard, calls)
            _collect_call_exprs(body, calls)
    elif isinstance(statement, Block):
        for body_statement in statement.body:
            _collect_call_exprs(body_statement, calls)
    elif isinstance(statement, FuncDef):
        _collect_call_exprs_from_func_item(statement, calls)
    elif isinstance(statement, OverloadedFuncDef):
        for item in statement.items:
            _collect_call_exprs(item, calls)
        if statement.impl is not None:
            _collect_call_exprs(statement.impl, calls)
    elif isinstance(statement, Decorator):
        _collect_call_exprs(statement.func, calls)
        for decorator in statement.decorators:
            _collect_call_exprs(decorator, calls)
    elif isinstance(statement, ClassDef):
        for decorator in statement.decorators:
            _collect_call_exprs(decorator, calls)
        for base_type_expression in statement.base_type_exprs:
            _collect_call_exprs(base_type_expression, calls)
        if statement.metaclass is not None:
            _collect_call_exprs(statement.metaclass, calls)
        for keyword_expression in statement.keywords.values():
            _collect_call_exprs(keyword_expression, calls)
        _collect_call_exprs(statement.defs, calls)
        if statement.analyzed is not None:
            _collect_call_exprs(statement.analyzed, calls)


def _collect_call_exprs_from_func_item(
    func_item: FuncItem,
    calls: list[CallExpr],
    /,
) -> None:
    """Collect call expressions from a function or lambda."""
    for argument in func_item.arguments:
        if argument.initializer is not None:
            _collect_call_exprs(argument.initializer, calls)
    _collect_call_exprs(func_item.body, calls)


def _collect_call_exprs_from_expression(  # noqa: C901, PLR0912, PLR0915  # pylint: disable=too-complex,too-many-branches,too-many-statements
    expression: Expression,
    calls: list[CallExpr],
    /,
) -> None:
    """Collect call expressions from an expression."""
    if isinstance(expression, (MemberExpr, YieldFromExpr)):
        _collect_call_exprs(expression.expr, calls)
    elif isinstance(expression, YieldExpr):
        if expression.expr is not None:
            _collect_call_exprs(expression.expr, calls)
    elif isinstance(expression, OpExpr):
        _collect_call_exprs(expression.left, calls)
        _collect_call_exprs(expression.right, calls)
        if expression.analyzed is not None:
            _collect_call_exprs(expression.analyzed, calls)
    elif isinstance(expression, ComparisonExpr):
        for operand in expression.operands:
            _collect_call_exprs(operand, calls)
    elif isinstance(expression, SliceExpr):
        if expression.begin_index is not None:
            _collect_call_exprs(expression.begin_index, calls)
        if expression.end_index is not None:
            _collect_call_exprs(expression.end_index, calls)
        if expression.stride is not None:
            _collect_call_exprs(expression.stride, calls)
    elif isinstance(expression, (CastExpr, AssertTypeExpr)):
        _collect_call_exprs(expression.expr, calls)
    elif isinstance(expression, RevealExpr):
        if expression.kind == REVEAL_TYPE and expression.expr is not None:
            _collect_call_exprs(expression.expr, calls)
    elif isinstance(expression, AssignmentExpr):
        _collect_call_exprs(expression.target, calls)
        _collect_call_exprs(expression.value, calls)
    elif isinstance(expression, UnaryExpr):
        _collect_call_exprs(expression.expr, calls)
    elif isinstance(expression, (ListExpr, TupleExpr)):
        for item in expression.items:
            _collect_call_exprs(item, calls)
    elif isinstance(expression, DictExpr):
        for key, value in expression.items:
            if key is not None:
                _collect_call_exprs(key, calls)
            _collect_call_exprs(value, calls)
    elif isinstance(expression, TemplateStrExpr):
        for template_item in expression.items:
            if isinstance(template_item, tuple):
                _collect_call_exprs(template_item[0], calls)
                if template_item[3] is not None:
                    _collect_call_exprs(template_item[3], calls)
            else:
                _collect_call_exprs(template_item, calls)
    elif isinstance(expression, SetExpr):
        for item in expression.items:
            _collect_call_exprs(item, calls)
    elif isinstance(expression, IndexExpr):
        _collect_call_exprs(expression.base, calls)
        _collect_call_exprs(expression.index, calls)
        if expression.analyzed is not None:
            _collect_call_exprs(expression.analyzed, calls)
    elif isinstance(expression, GeneratorExpr):
        for index, sequence, conditions in zip(
            expression.indices,
            expression.sequences,
            expression.condlists,
            strict=True,
        ):
            _collect_call_exprs(sequence, calls)
            _collect_call_exprs(index, calls)
            for condition in conditions:
                _collect_call_exprs(condition, calls)
        _collect_call_exprs(expression.left_expr, calls)
    elif isinstance(expression, DictionaryComprehension):
        for index, sequence, conditions in zip(
            expression.indices,
            expression.sequences,
            expression.condlists,
            strict=True,
        ):
            _collect_call_exprs(sequence, calls)
            _collect_call_exprs(index, calls)
            for condition in conditions:
                _collect_call_exprs(condition, calls)
        _collect_call_exprs(expression.key, calls)
        _collect_call_exprs(expression.value, calls)
    elif isinstance(expression, (ListComprehension, SetComprehension)):
        _collect_call_exprs(expression.generator, calls)
    elif isinstance(expression, ConditionalExpr):
        _collect_call_exprs(expression.cond, calls)
        _collect_call_exprs(expression.if_expr, calls)
        _collect_call_exprs(expression.else_expr, calls)
    elif isinstance(expression, TypeApplication):
        _collect_call_exprs(expression.expr, calls)
    elif isinstance(expression, LambdaExpr):
        _collect_call_exprs_from_func_item(expression, calls)
    elif isinstance(expression, (StarExpr, AwaitExpr)):
        _collect_call_exprs(expression.expr, calls)
    elif isinstance(expression, SuperExpr):
        _collect_call_exprs(expression.call, calls)


def _iter_call_exprs(node: Node, /) -> list[CallExpr]:
    """Return call expressions contained in a node."""
    calls: list[CallExpr] = []
    _collect_call_exprs(node, calls)
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
        match node:
            case FuncDef() | OverloadedFuncDef():
                fullname = node.fullname
                typ = node.type
                skip_bound_argument = node.has_self_or_cls_argument
            case Decorator():
                fullname = node.fullname
                typ = node.func.type
                skip_bound_argument = node.func.has_self_or_cls_argument
            case _:
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
    for expr in _iter_call_exprs(ctx.cls.defs):
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
