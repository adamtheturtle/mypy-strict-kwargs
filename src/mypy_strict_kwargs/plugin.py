"""``mypy`` plugin to enforce strict keyword arguments."""

import configparser
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NoReturn, assert_never

from mypy.errorcodes import CALL_ARG
from mypy.errors import CompileError
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
    NameExpr,
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
    SymbolNode,
    TemplateStrExpr,
    TryStmt,
    TupleExpr,
    TypeApplication,
    TypeInfo,
    UnaryExpr,
    Var,
    WhileStmt,
    WithStmt,
    YieldExpr,
    YieldFromExpr,
)
from mypy.options import Options
from mypy.patterns import (
    AsPattern,
    ClassPattern,
    MappingPattern,
    OrPattern,
    Pattern,
    SequencePattern,
    SingletonPattern,
    StarredPattern,
    ValuePattern,
)
from mypy.plugin import (
    ClassDefContext,
    FunctionSigContext,
    MethodSigContext,
    Plugin,
    ReportConfigContext,
)
from mypy.types import (
    CallableType,
    EllipsisType,
    FunctionLike,
    Type,
    UnboundType,
    UnpackType,
)

_CallExprContainer = Expression | Statement


class _CollectedCalls(list[CallExpr]):
    """Call expressions and fixed tuple lengths visible to each call."""

    def __init__(self) -> None:
        """Initialize an empty collection."""
        super().__init__()
        self.fixed_tuple_lengths: dict[CallExpr, dict[str, int]] = {}
        # Parameter names bound by scopes already attached to each call, so
        # that an inner scope's name shadows an outer scope's parameter of
        # the same name even when the inner binding is not a fixed tuple.
        self.bound_parameter_names: dict[CallExpr, set[str]] = {}


# Special methods which Python and ``mypy`` invoke implicitly, mapped to
# the number of arguments supplied by position at those implicit call
# sites.  Those arguments are never written by the caller, so making them
# keyword-only would make valid code impossible to express.
#
# Methods which are always invoked with no arguments, such as
# ``__len__``, have nothing to preserve and so are not listed.
_IMPLICIT_POSITIONAL_ARGUMENT_COUNTS = {
    # Callable objects and descriptors.
    "__call__": 1,
    "__delete__": 1,
    "__get__": 2,
    "__set__": 2,
    "__set_name__": 2,
    # Attribute access.
    "__delattr__": 1,
    "__getattr__": 1,
    "__getattribute__": 1,
    "__setattr__": 2,
    # Item access.
    "__class_getitem__": 1,
    "__contains__": 1,
    "__delitem__": 1,
    "__getitem__": 1,
    "__missing__": 1,
    "__setitem__": 2,
    # Context managers.
    "__aexit__": 3,
    "__exit__": 3,
    # Comparisons.
    "__eq__": 1,
    "__ge__": 1,
    "__gt__": 1,
    "__le__": 1,
    "__lt__": 1,
    "__ne__": 1,
    # Binary, reflected and in-place operators.
    "__add__": 1,
    "__and__": 1,
    "__divmod__": 1,
    "__floordiv__": 1,
    "__iadd__": 1,
    "__iand__": 1,
    "__ifloordiv__": 1,
    "__ilshift__": 1,
    "__imatmul__": 1,
    "__imod__": 1,
    "__imul__": 1,
    "__ior__": 1,
    "__ipow__": 2,
    "__irshift__": 1,
    "__isub__": 1,
    "__itruediv__": 1,
    "__ixor__": 1,
    "__lshift__": 1,
    "__matmul__": 1,
    "__mod__": 1,
    "__mul__": 1,
    "__or__": 1,
    "__pow__": 2,
    "__radd__": 1,
    "__rand__": 1,
    "__rdivmod__": 1,
    "__rfloordiv__": 1,
    "__rlshift__": 1,
    "__rmatmul__": 1,
    "__rmod__": 1,
    "__rmul__": 1,
    "__ror__": 1,
    "__rpow__": 1,
    "__rrshift__": 1,
    "__rshift__": 1,
    "__rsub__": 1,
    "__rtruediv__": 1,
    "__rxor__": 1,
    "__sub__": 1,
    "__truediv__": 1,
    "__xor__": 1,
    # Other implicitly invoked special methods.
    "__buffer__": 1,
    "__deepcopy__": 1,
    "__format__": 1,
    "__instancecheck__": 1,
    "__mro_entries__": 1,
    "__reduce_ex__": 1,
    "__release_buffer__": 1,
    "__round__": 1,
    "__setstate__": 1,
    "__subclasscheck__": 1,
    "__subclasshook__": 1,
}


def _preserved_positional_argument_count(
    ctx: FunctionSigContext | MethodSigContext,
    fullname: str,
) -> int:
    """Return positional arguments used by an implicit protocol
    operation.
    """
    if not isinstance(ctx, MethodSigContext):
        return 0

    method_name = fullname.rsplit(sep=".", maxsplit=1)[-1]
    preserved_count = _IMPLICIT_POSITIONAL_ARGUMENT_COUNTS.get(method_name, 0)
    if preserved_count == 0:
        return 0

    context = ctx.context
    if isinstance(context, CallExpr):
        context = context.callee
    if isinstance(context, MemberExpr) and context.name == method_name:
        # An explicit ``obj.__call__(...)``-style access.
        return 0
    return preserved_count


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
        preserved_positional_argument_count=(
            _preserved_positional_argument_count(ctx=ctx, fullname=fullname)
        ),
    )


def _transform_callable_type(
    *,
    signature: CallableType,
    fullname: str,
    ignore_names: list[str],
    skip_bound_argument: bool,
    preserved_positional_argument_count: int,
) -> CallableType:
    """Transform positional arguments in a callable type."""
    new_arg_kinds: list[ArgKind] = []

    star_arg_indices = [
        index
        for index, kind in enumerate(iterable=signature.arg_kinds)
        if kind == ArgKind.ARG_STAR
    ]

    first_star_arg_index = star_arg_indices[0] if star_arg_indices else None

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


def _super_method_mro(
    *,
    ctx: ClassDefContext,
    expr: CallExpr,
) -> list[TypeInfo]:
    """Return method-resolution entries searched by ``super()``."""
    callee = expr.callee
    if not isinstance(callee, SuperExpr) or not callee.call.args:
        return ctx.cls.info.mro[1:]

    explicit_super_type = callee.call.args[0]
    if not isinstance(
        explicit_super_type,
        (NameExpr, MemberExpr),
    ):
        return ctx.cls.info.mro[1:]

    symbol = ctx.api.lookup_qualified(
        name=explicit_super_type.name,
        ctx=explicit_super_type,
        suppress_errors=True,
    )
    explicit_super_info = None if symbol is None else symbol.node
    if not isinstance(explicit_super_info, TypeInfo):
        return ctx.cls.info.mro[1:]

    try:
        super_type_index = ctx.cls.info.mro.index(explicit_super_info)
    except ValueError:
        return ctx.cls.info.mro[1:]
    return ctx.cls.info.mro[super_type_index + 1 :]


def _known_sequence_length(
    *,
    expression: Expression,
    fixed_tuple_lengths: dict[str, int],
) -> int | None:
    """Return the length of a sequence expression, if it is known.

    ``None`` means that the length is not statically known.
    """
    match expression:
        case TupleExpr(items=items) | ListExpr(items=items):
            total = 0
            for item in items:
                if not isinstance(item, StarExpr):
                    total += 1
                    continue
                spread_length = _known_sequence_length(
                    expression=item.expr,
                    fixed_tuple_lengths=fixed_tuple_lengths,
                )
                if spread_length is None:
                    return None
                total += spread_length
            return total
        case NameExpr(name=name):
            return fixed_tuple_lengths.get(name)
        case _:
            return None


def _leading_positional_count(
    *,
    items: list[Expression],
    fixed_tuple_lengths: dict[str, int],
) -> int:
    """Return the number of items which occupy known positions.

    A ``*spread`` of unknown length inside a sequence literal makes every
    later item impossible to place, so counting stops there.  A spread whose
    length is known contributes that many positions.
    """
    leading_count = 0
    for item in items:
        if not isinstance(item, StarExpr):
            leading_count += 1
            continue
        spread_length = _known_sequence_length(
            expression=item.expr,
            fixed_tuple_lengths=fixed_tuple_lengths,
        )
        if spread_length is None:
            break
        leading_count += spread_length
    return leading_count


def _spread_positional_count(
    *,
    expression: Expression,
    fixed_tuple_lengths: dict[str, int],
) -> int:
    """Return positions known to be filled by a ``*`` argument."""
    match expression:
        case TupleExpr(items=items) | ListExpr(items=items):
            return _leading_positional_count(
                items=items,
                fixed_tuple_lengths=fixed_tuple_lengths,
            )
        case _:
            return (
                _known_sequence_length(
                    expression=expression,
                    fixed_tuple_lengths=fixed_tuple_lengths,
                )
                or 0
            )


def _call_disallows_positional_argument(
    *,
    call: CallExpr,
    signature: CallableType,
    fullname: str,
    ignore_names: list[str],
    skip_bound_argument: bool,
    fixed_tuple_lengths: dict[str, int],
) -> bool:
    """Return whether a call passes a transformed argument by position."""
    transformed = _transform_callable_type(
        signature=signature,
        fullname=fullname,
        ignore_names=ignore_names,
        skip_bound_argument=skip_bound_argument,
        preserved_positional_argument_count=0,
    )

    formal_arg_index = 1 if skip_bound_argument else 0
    for actual_arg, actual_arg_kind in zip(
        call.args,
        call.arg_kinds,
        strict=True,
    ):
        if actual_arg_kind == ArgKind.ARG_POS:
            positional_argument_count = 1
        elif actual_arg_kind == ArgKind.ARG_STAR:
            positional_argument_count = _spread_positional_count(
                expression=actual_arg,
                fixed_tuple_lengths=fixed_tuple_lengths,
            )
        else:
            continue

        for _ in range(positional_argument_count):
            if formal_arg_index >= len(transformed.arg_kinds):
                return False

            formal_arg_kind = transformed.arg_kinds[formal_arg_index]
            if formal_arg_kind == ArgKind.ARG_STAR:
                return False
            formal_arg_index += 1

            if formal_arg_kind in {
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
    fixed_tuple_lengths: dict[str, int],
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
            fixed_tuple_lengths=fixed_tuple_lengths,
        )
        for callable_item in callable_items
    )


def _collect_call_exprs(
    item: _CallExprContainer,
    calls: _CollectedCalls,
    /,
) -> None:
    """Collect call expressions from a syntax-tree node or expression."""
    match item:
        case CallExpr():
            calls.append(item)
            _collect_call_exprs(item.callee, calls)
            for argument in item.args:
                _collect_call_exprs(argument, calls)
            # ``analyzed`` holds the special-form rewrite of a call (such
            # as a ``cast()`` call).  It is populated during type
            # checking, which runs after this base-class hook, so it is
            # always ``None`` for the class body we traverse here.
            if item.analyzed is not None:  # pragma: no cover
                _collect_call_exprs(item.analyzed, calls)
        case Statement() as statement:
            _collect_call_exprs_from_statement(statement, calls)
        case Expression() as expression:
            _collect_call_exprs_from_expression(expression, calls)
        case _ as unreachable:
            assert_never(unreachable)


def _collect_call_exprs_from_statement(  # noqa: C901, PLR0912, PLR0915  # pylint: disable=too-complex,too-many-branches,too-many-statements
    statement: Statement,
    calls: _CollectedCalls,
    /,
) -> None:
    """Collect call expressions from a statement."""
    match statement:
        case ExpressionStmt(expr=expr):
            _collect_call_exprs(expr, calls)
        case AssignmentStmt(rvalue=rvalue, lvalues=lvalues):
            _collect_call_exprs(rvalue, calls)
            for lvalue in lvalues:
                _collect_call_exprs(lvalue, calls)
        case OperatorAssignmentStmt(rvalue=rvalue, lvalue=lvalue):
            _collect_call_exprs(rvalue, calls)
            _collect_call_exprs(lvalue, calls)
        case WhileStmt(expr=expr, body=body, else_body=else_body):
            _collect_call_exprs(expr, calls)
            _collect_call_exprs(body, calls)
            if else_body is not None:
                _collect_call_exprs(else_body, calls)
        case ForStmt(
            index=index,
            expr=expr,
            body=body,
            else_body=else_body,
        ):
            _collect_call_exprs(index, calls)
            _collect_call_exprs(expr, calls)
            _collect_call_exprs(body, calls)
            if else_body is not None:
                _collect_call_exprs(else_body, calls)
        case ReturnStmt(expr=expr):
            if expr is not None:
                _collect_call_exprs(expr, calls)
        case AssertStmt(expr=expr, msg=msg):
            _collect_call_exprs(expr, calls)
            if msg is not None:
                _collect_call_exprs(msg, calls)
        case DelStmt(expr=expr):
            _collect_call_exprs(expr, calls)
        case IfStmt(expr=conditions, body=body, else_body=else_body):
            for condition in conditions:
                _collect_call_exprs(condition, calls)
            for block in body:
                _collect_call_exprs(block, calls)
            if else_body is not None:
                _collect_call_exprs(else_body, calls)
        case RaiseStmt(expr=expr, from_expr=from_expr):
            if expr is not None:
                _collect_call_exprs(expr, calls)
            if from_expr is not None:
                _collect_call_exprs(from_expr, calls)
        case TryStmt(
            body=body,
            types=handler_types,
            handlers=handlers,
            vars=variables,
            else_body=else_body,
            finally_body=finally_body,
        ):
            _collect_call_exprs(body, calls)
            for handler_type, handler in zip(
                handler_types,
                handlers,
                strict=True,
            ):
                if handler_type is not None:
                    _collect_call_exprs(handler_type, calls)
                _collect_call_exprs(handler, calls)
            for variable in variables:
                if variable is not None:
                    _collect_call_exprs(variable, calls)
            if else_body is not None:
                _collect_call_exprs(else_body, calls)
            if finally_body is not None:
                _collect_call_exprs(finally_body, calls)
        case WithStmt(expr=expressions, target=targets, body=body):
            for expression, target in zip(
                expressions,
                targets,
                strict=True,
            ):
                _collect_call_exprs(expression, calls)
                if target is not None:
                    _collect_call_exprs(target, calls)
            _collect_call_exprs(body, calls)
        case MatchStmt(
            subject=subject,
            patterns=patterns,
            guards=guards,
            bodies=bodies,
        ):
            _collect_call_exprs(subject, calls)
            for pattern, guard, body in zip(
                patterns,
                guards,
                bodies,
                strict=True,
            ):
                _collect_call_exprs_from_pattern(pattern, calls)
                if guard is not None:
                    _collect_call_exprs(guard, calls)
                _collect_call_exprs(body, calls)
        case Block(body=body):
            for body_statement in body:
                _collect_call_exprs(body_statement, calls)
        case FuncDef():
            _collect_call_exprs_from_func_item(statement, calls)
        case OverloadedFuncDef(items=items):
            for overload_item in items:
                _collect_call_exprs(overload_item, calls)
        case Decorator(func=func, decorators=decorators):
            _collect_call_exprs(func, calls)
            for decorator in decorators:
                _collect_call_exprs(decorator, calls)
        case ClassDef(
            decorators=decorators,
            base_type_exprs=base_type_exprs,
            metaclass=metaclass,
            keywords=keywords,
        ):
            for decorator in decorators:
                _collect_call_exprs(decorator, calls)
            for base_type_expression in base_type_exprs:
                _collect_call_exprs(base_type_expression, calls)
            if metaclass is not None:
                _collect_call_exprs(metaclass, calls)
            for keyword_expression in keywords.values():
                _collect_call_exprs(keyword_expression, calls)
        case _:
            pass


def _collect_call_exprs_from_func_item(
    func_item: FuncItem,
    calls: _CollectedCalls,
    /,
) -> None:
    """Collect call expressions from a function or lambda."""
    for argument in func_item.arguments:
        if argument.initializer is not None:
            _collect_call_exprs(argument.initializer, calls)
    first_body_call_index = len(calls)
    _collect_call_exprs(func_item.body, calls)

    fixed_tuple_lengths = {
        argument.variable.name: fixed_tuple_length
        for argument in func_item.arguments
        if (
            fixed_tuple_length := _fixed_tuple_annotation_length(
                annotation=argument.type_annotation,
            )
        )
        is not None
    }
    for call in calls[first_body_call_index:]:
        # Inner scopes are collected first, so their parameters shadow this
        # enclosing scope's parameters of the same name -- even when the
        # inner binding is not a fixed tuple.
        bound_names = calls.bound_parameter_names.setdefault(call, set())
        lengths = calls.fixed_tuple_lengths.setdefault(call, {})
        for argument in func_item.arguments:
            name = argument.variable.name
            if name in bound_names:
                continue
            bound_names.add(name)
            if name in fixed_tuple_lengths:
                lengths[name] = fixed_tuple_lengths[name]


_TUPLE_ANNOTATION_NAMES = frozenset(
    {
        "Tuple",
        "builtins.tuple",
        "tuple",
        "typing.Tuple",
    }
)


def _unpacked_annotation(*, annotation: Type) -> Type | None:
    """Return the annotation unpacked by a PEP 646 ``*`` item.

    ``None`` means that the item is not an unpack.  Both the star syntax
    (``*tuple[int, int]``) and the ``Unpack[...]`` spelling are
    recognized.
    """
    if isinstance(annotation, UnpackType):
        return annotation.type
    if (
        isinstance(annotation, UnboundType)
        and annotation.name.rsplit(sep=".", maxsplit=1)[-1] == "Unpack"
        and len(annotation.args) == 1
    ):
        return annotation.args[0]
    return None


def _fixed_tuple_annotation_length(*, annotation: Type | None) -> int | None:
    """Return the length of a fixed tuple annotation, if known."""
    if (
        not isinstance(annotation, UnboundType)
        or annotation.name not in _TUPLE_ANNOTATION_NAMES
    ):
        return None
    if annotation.empty_tuple_index:
        return 0
    if not annotation.args or isinstance(annotation.args[-1], EllipsisType):
        return None

    length = 0
    for item in annotation.args:
        unpacked = _unpacked_annotation(annotation=item)
        if unpacked is None:
            length += 1
            continue
        unpacked_length = _fixed_tuple_annotation_length(annotation=unpacked)
        if unpacked_length is None:
            return None
        length += unpacked_length
    return length


def _collect_call_exprs_from_expression(  # noqa: C901, PLR0912, PLR0915  # pylint: disable=too-complex,too-many-branches,too-many-statements
    expression: Expression,
    calls: _CollectedCalls,
    /,
) -> None:
    """Collect call expressions from an expression."""
    match expression:
        case MemberExpr(expr=expr) | YieldFromExpr(expr=expr):
            _collect_call_exprs(expr, calls)
        case YieldExpr(expr=expr):
            if expr is not None:
                _collect_call_exprs(expr, calls)
        case OpExpr(left=left, right=right) as op_expr:
            _collect_call_exprs(left, calls)
            _collect_call_exprs(right, calls)
            # ``analyzed`` (e.g. a ``X | Y`` type expression) is only set
            # during type checking, after this base-class hook runs.
            if op_expr.analyzed is not None:  # pragma: no cover
                _collect_call_exprs(op_expr.analyzed, calls)
        case ComparisonExpr(operands=operands):
            for operand in operands:
                _collect_call_exprs(operand, calls)
        case SliceExpr(
            begin_index=begin_index,
            end_index=end_index,
            stride=stride,
        ):
            if begin_index is not None:
                _collect_call_exprs(begin_index, calls)
            if end_index is not None:
                _collect_call_exprs(end_index, calls)
            if stride is not None:
                _collect_call_exprs(stride, calls)
        # ``cast()``/``assert_type()``/``reveal_type()`` are rewritten
        # into these nodes during type checking, after this base-class
        # hook runs; in the class body we traverse they are still plain
        # ``CallExpr`` nodes, so these branches are never reached here.
        case (
            CastExpr(expr=expr) | AssertTypeExpr(expr=expr)
        ):  # pragma: no cover
            _collect_call_exprs(expr, calls)
        case RevealExpr(kind=kind, expr=expr):  # pragma: no cover
            if kind == REVEAL_TYPE and expr is not None:
                _collect_call_exprs(expr, calls)
        case AssignmentExpr(target=target, value=value):
            _collect_call_exprs(target, calls)
            _collect_call_exprs(value, calls)
        case UnaryExpr(expr=expr):
            _collect_call_exprs(expr, calls)
        case ListExpr(items=items) | TupleExpr(items=items):
            for item in items:
                _collect_call_exprs(item, calls)
        case DictExpr(items=items):
            for key, value in items:
                if key is not None:
                    _collect_call_exprs(key, calls)
                _collect_call_exprs(value, calls)
        # PEP 750 template strings (``t"..."``) only parse on Python
        # 3.14+, so this branch is not exercised by the test suite.
        case TemplateStrExpr(items=template_items):  # pragma: no cover
            for template_item in template_items:
                if isinstance(template_item, tuple):
                    expression, _, _, format_expr = template_item
                    _collect_call_exprs(expression, calls)
                    if format_expr is not None:
                        _collect_call_exprs(format_expr, calls)
                else:
                    _collect_call_exprs(template_item, calls)
        case SetExpr(items=items):
            for item in items:
                _collect_call_exprs(item, calls)
        case IndexExpr(base=base, index=index) as index_expr:
            _collect_call_exprs(base, calls)
            _collect_call_exprs(index, calls)
            # ``analyzed`` (a type application or type alias) is only set
            # during type checking, after this base-class hook runs.
            if index_expr.analyzed is not None:  # pragma: no cover
                _collect_call_exprs(index_expr.analyzed, calls)
        case GeneratorExpr(
            indices=indices,
            sequences=sequences,
            condlists=condlists,
            left_expr=left_expr,
        ):
            for index, sequence, conditions in zip(
                indices,
                sequences,
                condlists,
                strict=True,
            ):
                _collect_call_exprs(sequence, calls)
                _collect_call_exprs(index, calls)
                for condition in conditions:
                    _collect_call_exprs(condition, calls)
            _collect_call_exprs(left_expr, calls)
        case DictionaryComprehension(
            indices=indices,
            sequences=sequences,
            condlists=condlists,
            key=key,
            value=value,
        ):
            for index, sequence, conditions in zip(
                indices,
                sequences,
                condlists,
                strict=True,
            ):
                _collect_call_exprs(sequence, calls)
                _collect_call_exprs(index, calls)
                for condition in conditions:
                    _collect_call_exprs(condition, calls)
            _collect_call_exprs(key, calls)
            _collect_call_exprs(value, calls)
        case (
            ListComprehension(generator=generator)
            | SetComprehension(
                generator=generator,
            )
        ):
            _collect_call_exprs(generator, calls)
        case ConditionalExpr(cond=cond, if_expr=if_expr, else_expr=else_expr):
            _collect_call_exprs(cond, calls)
            _collect_call_exprs(if_expr, calls)
            _collect_call_exprs(else_expr, calls)
        # A bare ``TypeApplication`` only appears as the ``analyzed`` form
        # of an ``IndexExpr`` produced during type checking, after this
        # base-class hook runs, so this branch is never reached here.
        case TypeApplication(expr=expr):  # pragma: no cover
            _collect_call_exprs(expr, calls)
        case LambdaExpr():
            _collect_call_exprs_from_func_item(expression, calls)
        case StarExpr(expr=expr) | AwaitExpr(expr=expr):
            _collect_call_exprs(expr, calls)
        case SuperExpr(call=call):
            _collect_call_exprs(call, calls)
        case _:
            pass


def _collect_call_exprs_from_patterns(
    patterns: list[Pattern],
    calls: _CollectedCalls,
    /,
) -> None:
    """Collect call expressions from match patterns."""
    for pattern in patterns:
        _collect_call_exprs_from_pattern(pattern, calls)


def _collect_call_exprs_from_as_pattern(
    *,
    inner_pattern: Pattern | None,
    name: Expression | None,
    calls: _CollectedCalls,
) -> None:
    """Collect call expressions from an as pattern."""
    if inner_pattern is not None:
        _collect_call_exprs_from_pattern(inner_pattern, calls)
    if name is not None:
        _collect_call_exprs(name, calls)


def _collect_call_exprs_from_mapping_pattern(
    *,
    keys: list[Expression],
    values: list[Pattern],
    rest: Expression | None,
    calls: _CollectedCalls,
) -> None:
    """Collect call expressions from a mapping pattern."""
    for key in keys:
        _collect_call_exprs(key, calls)
    _collect_call_exprs_from_patterns(values, calls)
    if rest is not None:
        _collect_call_exprs(rest, calls)


def _collect_call_exprs_from_class_pattern(
    *,
    class_ref: Expression,
    positionals: list[Pattern],
    keyword_values: list[Pattern],
    calls: _CollectedCalls,
) -> None:
    """Collect call expressions from a class pattern."""
    _collect_call_exprs(class_ref, calls)
    _collect_call_exprs_from_patterns(positionals, calls)
    _collect_call_exprs_from_patterns(keyword_values, calls)


def _collect_call_exprs_from_pattern(
    pattern: Pattern,
    calls: _CollectedCalls,
    /,
) -> None:
    """Collect call expressions from a match pattern."""
    assert isinstance(  # noqa: S101
        pattern,
        (
            AsPattern,
            OrPattern,
            ValuePattern,
            SingletonPattern,
            SequencePattern,
            StarredPattern,
            MappingPattern,
            ClassPattern,
        ),
    )
    match pattern:
        case AsPattern(pattern=inner_pattern, name=name):
            _collect_call_exprs_from_as_pattern(
                inner_pattern=inner_pattern,
                name=name,
                calls=calls,
            )
        case OrPattern(patterns=patterns) | SequencePattern(patterns=patterns):
            _collect_call_exprs_from_patterns(patterns, calls)
        case ValuePattern(expr=expr):
            _collect_call_exprs(expr, calls)
        case StarredPattern(capture=capture):
            if capture is not None:
                _collect_call_exprs(capture, calls)
        case SingletonPattern():
            pass
        case MappingPattern(keys=keys, values=values, rest=rest):
            _collect_call_exprs_from_mapping_pattern(
                keys=keys,
                values=values,
                rest=rest,
                calls=calls,
            )
        case ClassPattern(
            class_ref=class_ref,
            positionals=positionals,
            keyword_values=keyword_values,
        ):
            _collect_call_exprs_from_class_pattern(
                class_ref=class_ref,
                positionals=positionals,
                keyword_values=keyword_values,
                calls=calls,
            )
        case _ as unreachable:
            assert_never(unreachable)


def _iter_call_exprs(node: _CallExprContainer, /) -> _CollectedCalls:
    """Return call expressions contained in a node."""
    calls = _CollectedCalls()
    _collect_call_exprs(node, calls)
    return calls


def _assigned_staticmethod_target(
    *,
    class_def: ClassDef,
    method_name: str,
) -> FuncDef | OverloadedFuncDef | Decorator | None:
    """Return the callable wrapped by an assigned ``staticmethod``."""
    for statement in class_def.defs.body:
        match statement:
            case AssignmentStmt(
                lvalues=[NameExpr(name=name)],
                rvalue=CallExpr(
                    callee=NameExpr(fullname="builtins.staticmethod"),
                    args=[
                        NameExpr(
                            node=(
                                FuncDef() | OverloadedFuncDef() | Decorator()
                            ) as target
                        )
                    ],
                ),
            ) if name == method_name:
                return target
            case _:
                continue
    return None


@dataclass(frozen=True, kw_only=True)
class _ResolvedSuperMember:
    """A resolved ``super()`` member node and its assigned full name."""

    node: FuncDef | OverloadedFuncDef | Decorator | None
    assigned_fullname: str | None


def _resolved_super_member_node(
    *,
    node: SymbolNode | None,
    class_def: ClassDef,
    method_name: str,
) -> _ResolvedSuperMember:
    """Resolve a supported member node and its assigned full name."""
    if isinstance(node, Var):
        return _ResolvedSuperMember(
            node=_assigned_staticmethod_target(
                class_def=class_def,
                method_name=method_name,
            ),
            assigned_fullname=node.fullname,
        )
    if isinstance(node, FuncDef | OverloadedFuncDef | Decorator):
        return _ResolvedSuperMember(node=node, assigned_fullname=None)
    return _ResolvedSuperMember(node=None, assigned_fullname=None)


def _check_super_method_call(
    *,
    ctx: ClassDefContext,
    expr: CallExpr,
    method_name: str,
    ignore_names: list[str],
    fixed_tuple_lengths: dict[str, int],
) -> None:
    """Check one ``super()`` method call expression."""
    for info in _super_method_mro(ctx=ctx, expr=expr):
        symbol = info.names.get(method_name)
        if symbol is None:
            continue

        resolved = _resolved_super_member_node(
            node=symbol.node,
            class_def=info.defn,
            method_name=method_name,
        )
        assigned_fullname = resolved.assigned_fullname

        match resolved.node:
            case FuncDef() | OverloadedFuncDef() as node:
                fullname = assigned_fullname or node.fullname
                typ = node.type
                skip_bound_argument = (
                    assigned_fullname is None and node.has_self_or_cls_argument
                )
            case Decorator() as node:
                fullname = assigned_fullname or node.fullname
                typ = node.func.type
                skip_bound_argument = (
                    assigned_fullname is None
                    and node.func.has_self_or_cls_argument
                )
            case _:
                return

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
            fixed_tuple_lengths=fixed_tuple_lengths,
        ):
            ctx.api.fail(
                msg=(
                    f'Too many positional arguments for "{method_name}" '
                    f'of "{info.name}"'
                ),
                ctx=expr,
                code=CALL_ARG,
            )
            return
        return


def _check_super_method_calls(
    ctx: ClassDefContext,
    *,
    ignore_names: list[str],
) -> None:
    """Check ``super()`` method calls in a class body.

    Needed because ``mypy`` does not run method signature hooks for
    ``super().method(...)``
    (https://github.com/python/mypy/issues/21744).
    """
    calls = _iter_call_exprs(ctx.cls.defs)
    for expr in calls:
        method_name = _super_method_name(expr=expr)
        if method_name is None:
            continue
        _check_super_method_call(
            ctx=ctx,
            expr=expr,
            method_name=method_name,
            ignore_names=ignore_names,
            fixed_tuple_lengths=calls.fixed_tuple_lengths.get(expr, {}),
        )


@dataclass(frozen=True, kw_only=True)
class _PluginConfiguration:
    """Validated ``mypy_strict_kwargs`` configuration."""

    ignore_names: list[str]
    debug: bool


def _config_error(
    *, config_file: Path, section: str, message: str
) -> NoReturn:
    """Raise a ``mypy`` configuration error for a plugin option."""
    raise CompileError(messages=[f"{config_file}: [{section}]: {message}"])


def _is_list(value: object, /) -> bool:
    """Return whether a configuration value is a list.

    This deliberately returns a plain ``bool`` rather than a
    ``TypeGuard`` so that callers keep their explicitly dynamic type
    instead of narrowing to a container of unknown items.
    """
    return isinstance(value, list)


def _is_table(value: object, /) -> bool:
    """Return whether a configuration value is a table.

    See ``_is_list`` for why this is not a ``TypeGuard``.
    """
    return isinstance(value, dict)


def _validated_ignore_names(
    *,
    value: object,
    config_file: Path,
    section: str,
) -> list[str]:
    """Return ``ignore_names`` after checking that it is a list of
    strings.
    """
    message = '"ignore_names" must be an array of strings'
    items: Any = value
    if not _is_list(items):
        _config_error(
            config_file=config_file,
            section=section,
            message=message,
        )

    ignore_names: list[str] = []
    for item in items:
        if not isinstance(item, str):
            _config_error(
                config_file=config_file,
                section=section,
                message=message,
            )
        ignore_names.append(item)
    return ignore_names


def _validated_debug(
    *,
    value: object,
    config_file: Path,
    section: str,
) -> bool:
    """Return ``debug`` after checking that it is a boolean."""
    if not isinstance(value, bool):
        _config_error(
            config_file=config_file,
            section=section,
            message='"debug" must be a boolean',
        )
    return value


def _toml_plugin_configuration(
    *,
    config_file: Path,
) -> _PluginConfiguration:
    """Return the plugin configuration from a TOML configuration file."""
    section = "tool.mypy_strict_kwargs"
    with config_file.open(mode="rb") as config_file_object:
        config_dictionary = tomllib.load(config_file_object)

    tools: dict[str, Any] = config_dictionary.get("tool", {})
    plugin_config: Any = tools.get("mypy_strict_kwargs", {})
    if not _is_table(plugin_config):
        _config_error(
            config_file=config_file,
            section=section,
            message="expected a table",
        )

    return _PluginConfiguration(
        ignore_names=_validated_ignore_names(
            value=plugin_config.get("ignore_names", []),
            config_file=config_file,
            section=section,
        ),
        debug=_validated_debug(
            value=plugin_config.get("debug", False),
            config_file=config_file,
            section=section,
        ),
    )


def _ini_plugin_configuration(
    *,
    config_file: Path,
) -> _PluginConfiguration:
    """Return the plugin configuration from an INI configuration file.

    This handles ``mypy.ini``, ``.mypy.ini`` and ``setup.cfg``.
    """
    section = "mypy_strict_kwargs"
    parser = configparser.ConfigParser()
    parser.read(filenames=config_file)

    if not parser.has_section(section=section):
        return _PluginConfiguration(ignore_names=[], debug=False)

    ignore_names_str = parser.get(
        section=section,
        option="ignore_names",
        fallback="",
    )
    ignore_names = [
        name.strip()
        for name in ignore_names_str.split(sep=",")
        if name.strip()
    ]

    try:
        debug = parser.getboolean(
            section=section,
            option="debug",
            fallback=False,
        )
    except ValueError:
        _config_error(
            config_file=config_file,
            section=section,
            message='"debug" must be a boolean',
        )

    return _PluginConfiguration(ignore_names=ignore_names, debug=debug)


def _plugin_configuration(
    *,
    config_file_path: str | None,
) -> _PluginConfiguration:
    """Return the validated configuration for the plugin."""
    if config_file_path is None:
        return _PluginConfiguration(ignore_names=[], debug=False)

    config_file = Path(config_file_path)
    if config_file.suffix == ".toml":
        return _toml_plugin_configuration(config_file=config_file)
    return _ini_plugin_configuration(config_file=config_file)


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
        configuration = _plugin_configuration(
            config_file_path=options.config_file,
        )
        self._ignore_names = configuration.ignore_names
        self._debug = configuration.debug

    def report_config_data(self, ctx: ReportConfigContext) -> object:
        """Return plugin configuration that affects cached modules."""
        del ctx
        return {
            "debug": self._debug,
            "ignore_names": self._ignore_names,
        }

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

        ``get_method_signature_hook`` is not invoked for
        ``super().method(...)``
        (https://github.com/python/mypy/issues/21744), so this hook
        walks class bodies and checks those call sites manually.
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
