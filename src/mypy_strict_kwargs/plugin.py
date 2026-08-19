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
from mypy.exprtotype import TypeTranslationError, expr_to_unanalyzed_type
from mypy.nodes import (
    REVEAL_TYPE,
    ArgKind,
    AssertStmt,
    AssertTypeExpr,
    AssignmentExpr,
    AssignmentStmt,
    AwaitExpr,
    Block,
    BytesExpr,
    CallExpr,
    CastExpr,
    ClassDef,
    ComparisonExpr,
    ConditionalExpr,
    Context,
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
    GlobalDecl,
    IfStmt,
    Import,
    ImportFrom,
    IndexExpr,
    IntExpr,
    LambdaExpr,
    ListComprehension,
    ListExpr,
    MatchStmt,
    MemberExpr,
    MypyFile,
    NameExpr,
    NonlocalDecl,
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
    StrExpr,
    SuperExpr,
    SymbolNode,
    TemplateStrExpr,
    TryStmt,
    TupleExpr,
    TypeAlias,
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
    AttributeContext,
    ClassDefContext,
    FunctionSigContext,
    MethodSigContext,
    Plugin,
    ReportConfigContext,
    SemanticAnalyzerPluginInterface,
)
from mypy.types import (
    CallableType,
    EllipsisType,
    FunctionLike,
    Instance,
    TupleType,
    Type,
    TypeVarTupleType,
    UnboundType,
    UnionType,
    UnpackType,
    get_proper_type,
)

_CallExprContainer = Expression | Statement


@dataclass(kw_only=True)
class _Scope:
    """Names bound while collecting calls from one Python scope."""

    names: set[str]
    is_comprehension: bool
    # Names this scope declares ``nonlocal``, which belong to an
    # enclosing function scope rather than to this one.
    nonlocal_names: set[str]
    # Names assigned a sequence of known length, mapped to that length.
    # ``None`` marks a name whose length is not known everywhere.
    assigned_lengths: dict[str, int | None]
    # Annotations of the parameters this scope introduces.
    annotations: dict[str, Type]


class _CollectedCalls(list[CallExpr]):
    """Call expressions and fixed tuple lengths visible to each call."""

    def __init__(self, *, resolver: "_AnnotationResolver") -> None:
        """Initialize an empty collection."""
        super().__init__()
        self.resolver = resolver
        self.fixed_tuple_lengths: dict[CallExpr, dict[str, int]] = {}
        # Names bound by scopes already attached to each call, so that an
        # inner scope's binding shadows an outer scope's parameter of the
        # same name even when the inner binding is not a fixed tuple.
        self.bound_names: dict[CallExpr, set[str]] = {}
        self._scopes: list[_Scope] = [
            _Scope(
                names=set(),
                is_comprehension=False,
                nonlocal_names=set(),
                assigned_lengths={},
                annotations={},
            )
        ]

    def push_scope(
        self,
        *,
        is_comprehension: bool,
        annotations: dict[str, Type],
    ) -> None:
        """Start collecting the names bound by a new scope."""
        self._scopes.append(
            _Scope(
                names=set(),
                is_comprehension=is_comprehension,
                nonlocal_names=set(),
                assigned_lengths={},
                annotations=annotations,
            )
        )

    def visible_lengths(self) -> dict[str, int]:
        """Return the known sequence lengths of visible names."""
        lengths: dict[str, int] = {}
        for scope in self._scopes:
            for name, length in scope.assigned_lengths.items():
                if length is None:
                    lengths.pop(name, None)
                else:
                    lengths[name] = length
            for name, annotation in scope.annotations.items():
                annotation_length = _fixed_tuple_annotation_length(
                    annotation=annotation,
                    resolver=self.resolver,
                )
                if annotation_length is None:
                    lengths.pop(name, None)
                else:
                    lengths[name] = annotation_length
        return lengths

    def annotation(self, name: str, /) -> Type | None:
        """Return the annotation of a name visible where it is used.

        A scope which binds the name without annotating it hides an
        enclosing scope's annotation.
        """
        for scope in reversed(self._scopes):
            if name in scope.annotations:
                return scope.annotations[name]
            if name in scope.names:
                return None
        return None

    def pop_scope(self) -> _Scope:
        """Finish a scope and return what it binds."""
        return self._scopes.pop()

    def declare_nonlocal(self, names: set[str], /) -> None:
        """Record names this scope declares ``nonlocal``.

        A ``nonlocal`` declaration binds nothing by itself; it says that
        assignments belong to an enclosing function scope.
        """
        self._scopes[-1].nonlocal_names |= names

    def bind(self, names: set[str], /) -> None:
        """Record names bound by the scope being collected."""
        for name in names:
            self.bind_length(name=name, length=None)

    def bind_length(self, *, name: str, length: int | None) -> None:
        """Record the length a name is assigned in this scope.

        A name assigned more than once keeps a length only when every
        assignment agrees.
        """
        for scope in self._bound_scopes(name=name):
            scope.names.add(name)
            if name in scope.assigned_lengths:
                existing = scope.assigned_lengths[name]
                scope.assigned_lengths[name] = (
                    length if existing == length else None
                )
                continue
            scope.assigned_lengths[name] = length

    def _bound_scopes(self, *, name: str) -> list[_Scope]:
        """Return the scopes an assignment to a name binds in.

        An assignment to a ``nonlocal`` name belongs to the nearest
        enclosing function scope which binds that name.  Scopes beyond
        that one keep what they knew, because ``nonlocal`` never reaches
        them.  A name no enclosing scope binds yet may be an enclosing
        parameter, which is not recorded as a binding, so every enclosing
        scope loses what it knew.
        """
        if name not in self._scopes[-1].nonlocal_names:
            return [self._scopes[-1]]
        enclosing = [
            scope for scope in self._scopes if not scope.is_comprehension
        ][:-1]
        bound: list[_Scope] = []
        for scope in reversed(enclosing):
            bound.append(scope)
            if name in scope.names:
                break
        return bound

    def bind_in_function_scope(self, names: set[str], /) -> None:
        """Record names bound in the nearest enclosing function scope.

        An assignment expression inside a comprehension binds in the
        scope containing the comprehension, not in the comprehension.
        """
        function_scopes = [
            scope for scope in self._scopes if not scope.is_comprehension
        ]
        function_scopes[-1].names |= names


def _binding_target_names(target: Expression, /) -> set[str]:
    """Return the names bound by an assignment or loop target."""
    match target:
        case NameExpr(name=name):
            return {name}
        case TupleExpr(items=items) | ListExpr(items=items):
            names: set[str] = set()
            for item in items:
                names |= _binding_target_names(item)
            return names
        case StarExpr(expr=expr):
            return _binding_target_names(expr)
        case _:
            return set()


def _patterns_bound_names(patterns: list[Pattern], /) -> set[str]:
    """Return the names captured by a list of match patterns."""
    names: set[str] = set()
    for pattern in patterns:
        names |= _pattern_bound_names(pattern)
    return names


def _as_pattern_bound_names(
    *,
    inner_pattern: Pattern | None,
    name: Expression | None,
) -> set[str]:
    """Return the names captured by an as pattern."""
    names: set[str] = set()
    if inner_pattern is not None:
        names |= _pattern_bound_names(inner_pattern)
    if name is not None:
        names |= _binding_target_names(name)
    return names


def _mapping_pattern_bound_names(
    *,
    values: list[Pattern],
    rest: Expression | None,
) -> set[str]:
    """Return the names captured by a mapping pattern."""
    names = _patterns_bound_names(values)
    if rest is not None:
        names |= _binding_target_names(rest)
    return names


def _pattern_bound_names(pattern: Pattern, /) -> set[str]:
    """Return the names captured by a match pattern."""
    match pattern:
        case AsPattern(pattern=inner_pattern, name=name):
            return _as_pattern_bound_names(
                inner_pattern=inner_pattern,
                name=name,
            )
        case OrPattern(patterns=patterns) | SequencePattern(patterns=patterns):
            return _patterns_bound_names(patterns)
        case StarredPattern(capture=capture):
            return set() if capture is None else _binding_target_names(capture)
        case MappingPattern(values=values, rest=rest):
            return _mapping_pattern_bound_names(values=values, rest=rest)
        case ClassPattern(positionals=positionals, keyword_values=keywords):
            return _patterns_bound_names([*positionals, *keywords])
        case _:
            return set()


def _apply_scope_to_calls(
    *,
    calls: _CollectedCalls,
    first_call_index: int,
    names: set[str],
    fixed_tuple_lengths: dict[str, int],
) -> None:
    """Attach the bindings of one scope to the calls it contains.

    Inner scopes are collected first, so a name they already bound is
    left alone: the nearest binding wins.
    """
    for call in calls[first_call_index:]:
        bound_names = calls.bound_names.setdefault(call, set())
        lengths = calls.fixed_tuple_lengths.setdefault(call, {})
        for name in names - bound_names:
            bound_names.add(name)
            if name in fixed_tuple_lengths:
                lengths[name] = fixed_tuple_lengths[name]


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
    "__rpow__": 2,
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


def _write_debug_fullname(*, fullname: str, path: str) -> None:
    """Write a full name which ``ignore_names`` accepts.

    Names found while checking a stub are not written: they are
    internal to the type stubs rather than names from the checked
    project, and they drown out the names a user is looking for.
    """
    if path.endswith(".pyi"):
        return
    sys.stderr.write(f"DEBUG: mypy_strict_kwargs: {fullname}\n")


# ``functools.partial`` and ``functools.partialmethod`` forward their
# extra positional arguments to the callable they wrap, so those
# arguments are checked against that callable rather than against the
# signature of ``partial`` itself.
_PARTIAL_FULLNAMES = frozenset(
    {"functools.partial", "functools.partialmethod"}
)


def _callable_description(*, name: str) -> str:
    """Return the way ``mypy`` names a callable in an error message."""
    parts = name.split(sep=" of ", maxsplit=1)
    if len(parts) == 1:
        return f'"{parts[0]}"'
    return f'"{parts[0]}" of "{parts[1]}"'


def _check_partial_arguments(
    *,
    ctx: FunctionSigContext | MethodSigContext,
    fullname: str,
    ignore_names: list[str],
) -> None:
    """Report parameters which ``partial`` binds by position."""
    # Pad so that a signature with fewer formal arguments than expected
    # cannot raise.
    argument_groups = [*ctx.args, [], []]
    wrapped_arguments, bound_arguments = argument_groups[0], argument_groups[1]
    if not wrapped_arguments or not bound_arguments:
        return

    wrapped = get_proper_type(
        typ=ctx.api.get_expression_type(node=wrapped_arguments[0])
    )
    if not isinstance(wrapped, CallableType):
        return
    if wrapped.name is None or wrapped.definition is None:
        return

    # ``partialmethod`` binds arguments after the ``self`` parameter,
    # which the descriptor supplies later.
    skip_bound_argument = fullname == "functools.partialmethod"
    transformed = _transform_callable_type(
        signature=wrapped,
        fullname=wrapped.definition.fullname,
        ignore_names=ignore_names,
        skip_bound_argument=skip_bound_argument,
        preserved_positional_argument_count=0,
    )
    if _formals_disallow_positional(
        transformed=transformed,
        first_formal_index=1 if skip_bound_argument else 0,
        positional_argument_count=len(bound_arguments),
    ):
        description = _callable_description(name=wrapped.name)
        ctx.api.fail(
            f"Too many positional arguments for {description}",
            ctx.context,
            code=CALL_ARG,
        )


def _transform_signature(
    ctx: FunctionSigContext | MethodSigContext,
    fullname: str,
    *,
    ignore_names: list[str],
    debug: bool,
) -> CallableType:
    """Transform positional arguments to keyword-only arguments."""
    if debug:
        _write_debug_fullname(fullname=fullname, path=ctx.api.path)

    if fullname in _PARTIAL_FULLNAMES:
        _check_partial_arguments(
            ctx=ctx,
            fullname=fullname,
            ignore_names=ignore_names,
        )

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


def _dotted_name(*, expression: Expression) -> str | None:
    """Return the dotted name an expression spells, if it is one."""
    match expression:
        case NameExpr(name=name):
            return name
        case MemberExpr(expr=inner_expression, name=name):
            prefix = _dotted_name(expression=inner_expression)
            return None if prefix is None else f"{prefix}.{name}"
        case _:
            return None


def _aliased_class_info(
    *,
    api: SemanticAnalyzerPluginInterface,
    var: Var,
    visited: frozenset[str],
) -> TypeInfo | None:
    """Return the class a module-level alias variable refers to.

    ``mypy`` records ``Alias: Final = SomeClass`` as a variable rather
    than as a type alias, so the assignment is looked up in the module
    which defines it.
    """
    module_name, _, variable_name = var.fullname.rpartition(".")
    module = api.modules.get(module_name)
    if module is None:
        return None
    for statement in module.defs:
        match statement:
            case AssignmentStmt(
                lvalues=[NameExpr(name=lvalue_name)],
                rvalue=rvalue,
            ) if lvalue_name == variable_name:
                return _named_class_info(
                    api=api,
                    expression=rvalue,
                    visited=visited,
                )
            case _:
                continue
    return None


def _named_class_info(
    *,
    api: SemanticAnalyzerPluginInterface,
    expression: Expression,
    visited: frozenset[str],
) -> TypeInfo | None:
    """Return the class an expression names, if it names one.

    ``visited`` holds the aliases already followed, so that a cyclic
    alias definition does not loop forever.
    """
    name = _dotted_name(expression=expression)
    if name is None:
        return None
    symbol = api.lookup_qualified(
        name=name,
        ctx=expression,
        suppress_errors=True,
    )
    node = None if symbol is None else symbol.node
    if isinstance(node, TypeInfo):
        return node
    if isinstance(node, TypeAlias):
        target = get_proper_type(typ=node.target)
        return target.type if isinstance(target, Instance) else None
    if isinstance(node, Var) and node.fullname not in visited:
        return _aliased_class_info(
            api=api,
            var=node,
            visited=visited | {node.fullname},
        )
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

    explicit_super_info = _named_class_info(
        api=ctx.api,
        expression=callee.call.args[0],
        visited=frozenset(),
    )
    if explicit_super_info is None:
        return ctx.cls.info.mro[1:]

    try:
        super_type_index = ctx.cls.info.mro.index(explicit_super_info)
    except ValueError:
        return ctx.cls.info.mro[1:]
    return ctx.cls.info.mro[super_type_index + 1 :]


def _known_items_length(
    *,
    items: list[Expression],
    fixed_tuple_lengths: dict[str, int],
) -> int | None:
    """Return the number of values a sequence literal holds, if known."""
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


def _known_bytes_length(*, value: str) -> int | None:
    """Return the number of bytes a bytes literal holds, if known.

    ``mypy`` stores a bytes literal as text with non-printable bytes
    escaped, so a value containing a backslash has no reliable length.
    """
    if "\\" in value:
        return None
    return len(value)


def _known_dict_length(
    *,
    items: list[tuple[Expression | None, Expression]],
) -> int | None:
    """Return the number of keys a dictionary literal holds, if known.

    Unpacking a dictionary yields its keys.  A ``**spread`` entry has a
    ``None`` key and contributes an unknown number of keys.
    """
    keys = [key for key, _ in items]
    if None in keys:
        return None
    return len(keys)


def _selected_item(
    *,
    base: Expression,
    index: Expression,
) -> Expression | None:
    """Return the item a constant subscript selects from a literal."""
    match (base, index):
        case (
            (TupleExpr(items=items) | ListExpr(items=items)),
            IntExpr(value=position),
        ) if -len(items) <= position < len(items):
            return items[position]
        case (DictExpr(items=entries), IntExpr(value=key_value)):
            for key, value in entries:
                if isinstance(key, IntExpr) and key.value == key_value:
                    return value
        case (DictExpr(items=entries), StrExpr(value=key_text)):
            for key, value in entries:
                if isinstance(key, StrExpr) and key.value == key_text:
                    return value
        case _:
            return None
    return None


def _known_slice_length(
    *,
    base: Expression,
    index: SliceExpr,
    fixed_tuple_lengths: dict[str, int],
) -> int | None:
    """Return the length of a constant slice of a sequence literal."""
    del fixed_tuple_lengths
    if not isinstance(base, TupleExpr | ListExpr):
        return None
    if any(isinstance(item, StarExpr) for item in base.items):
        # A spread makes the positions of the items unknown.
        return None

    bounds: list[int | None] = []
    for bound in (index.begin_index, index.end_index):
        if bound is None:
            bounds.append(None)
        elif isinstance(bound, IntExpr):
            bounds.append(bound.value)
        else:
            return None
    if index.stride is not None:
        return None
    return len(base.items[bounds[0] : bounds[1]])


def _known_subscript_length(
    *,
    expression: IndexExpr,
    fixed_tuple_lengths: dict[str, int],
) -> int | None:
    """Return the length of a constant subscript, if it is known."""
    if isinstance(expression.index, SliceExpr):
        return _known_slice_length(
            base=expression.base,
            index=expression.index,
            fixed_tuple_lengths=fixed_tuple_lengths,
        )
    item = _selected_item(base=expression.base, index=expression.index)
    if item is None:
        return None
    return _known_sequence_length(
        expression=item,
        fixed_tuple_lengths=fixed_tuple_lengths,
    )


def _known_operator_length(
    *,
    expression: OpExpr,
    fixed_tuple_lengths: dict[str, int],
) -> int | None:
    """Return the length an operator expression produces, if known."""
    left = _known_sequence_length(
        expression=expression.left,
        fixed_tuple_lengths=fixed_tuple_lengths,
    )
    if left is None:
        return None

    if expression.op in {"and", "or"}:
        # An empty sequence is the only sequence of known length which
        # is false in a boolean context.
        takes_left = (left == 0) if expression.op == "and" else (left != 0)
        if takes_left:
            return left
        return _known_sequence_length(
            expression=expression.right,
            fixed_tuple_lengths=fixed_tuple_lengths,
        )

    if expression.op == "+":
        right = _known_sequence_length(
            expression=expression.right,
            fixed_tuple_lengths=fixed_tuple_lengths,
        )
        return None if right is None else left + right

    if expression.op == "*" and isinstance(expression.right, IntExpr):
        return left * max(expression.right.value, 0)
    return None


def _known_branch_length(
    *,
    branches: tuple[Expression, Expression],
    fixed_tuple_lengths: dict[str, int],
) -> int | None:
    """Return the length both branches produce, if they agree."""
    lengths = {
        _known_sequence_length(
            expression=branch,
            fixed_tuple_lengths=fixed_tuple_lengths,
        )
        for branch in branches
    }
    return lengths.pop() if len(lengths) == 1 else None


def _known_literal_length(
    *,
    expression: Expression,
    fixed_tuple_lengths: dict[str, int],
) -> int | None:
    """Return the length of a literal or a name, if it is known."""
    match expression:
        case (
            TupleExpr(items=items)
            | ListExpr(items=items)
            | SetExpr(items=items)
        ):
            return _known_items_length(
                items=items,
                fixed_tuple_lengths=fixed_tuple_lengths,
            )
        case DictExpr(items=dict_items):
            return _known_dict_length(items=dict_items)
        case StrExpr(value=value):
            return len(value)
        case BytesExpr(value=value):
            return _known_bytes_length(value=value)
        case NameExpr(name=name):
            return fixed_tuple_lengths.get(name)
        case _:
            return None


def _known_sequence_length(
    *,
    expression: Expression,
    fixed_tuple_lengths: dict[str, int],
) -> int | None:
    """Return the length of a sequence expression, if it is known.

    ``None`` means that the length is not statically known.
    """
    match expression:
        case AssignmentExpr(value=assigned_value):
            return _known_sequence_length(
                expression=assigned_value,
                fixed_tuple_lengths=fixed_tuple_lengths,
            )
        case ConditionalExpr(if_expr=if_expr, else_expr=else_expr):
            return _known_branch_length(
                branches=(if_expr, else_expr),
                fixed_tuple_lengths=fixed_tuple_lengths,
            )
        case IndexExpr() as index_expression:
            return _known_subscript_length(
                expression=index_expression,
                fixed_tuple_lengths=fixed_tuple_lengths,
            )
        case OpExpr() as operator_expression:
            return _known_operator_length(
                expression=operator_expression,
                fixed_tuple_lengths=fixed_tuple_lengths,
            )
        case _:
            return _known_literal_length(
                expression=expression,
                fixed_tuple_lengths=fixed_tuple_lengths,
            )


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


_ASSERT_TYPE_FULLNAMES = frozenset(
    {"typing.assert_type", "typing_extensions.assert_type"}
)
_CAST_FULLNAMES = frozenset({"typing.cast", "typing_extensions.cast"})
_GET_ITEM_FULLNAMES = frozenset(
    {
        "_operator.__getitem__",
        "_operator.getitem",
        "operator.__getitem__",
        "operator.getitem",
    }
)


def _returned_annotation(
    *,
    node: SymbolNode | None,
) -> Type | None:
    """Return the declared return type of a callable definition."""
    match node:
        case FuncDef() as function:
            signature = function.type
        case Decorator() as decorator:
            signature = decorator.func.type
        case _:
            return None
    if not isinstance(signature, CallableType):
        return None
    return signature.ret_type


def _annotation_from_expression(
    *,
    expression: Expression,
    resolver: "_AnnotationResolver",
) -> Type | None:
    """Return the annotation an expression spells, if it spells one."""
    try:
        return expr_to_unanalyzed_type(
            expr=expression,
            options=resolver.api.options,
        )
    except TypeTranslationError:  # pragma: no cover
        # An invalid ``cast()`` target is reported by the type checker
        # itself, so this only guards against a malformed syntax tree.
        return None


def _called_expression_length(
    *,
    expression: CallExpr,
    fixed_tuple_lengths: dict[str, int],
    resolver: "_AnnotationResolver",
    class_info: TypeInfo,
) -> int | None:
    """Return the length a call is known to produce."""
    name = _dotted_name(expression=expression.callee)
    if name is None:
        return None
    fullname = resolver.fullname(name)
    arguments = expression.args

    if fullname in _ASSERT_TYPE_FULLNAMES and arguments:
        return _expression_length(
            expression=arguments[0],
            fixed_tuple_lengths=fixed_tuple_lengths,
            resolver=resolver,
            class_info=class_info,
        )
    if fullname in _CAST_FULLNAMES and arguments:
        return _fixed_tuple_annotation_length(
            annotation=_annotation_from_expression(
                expression=arguments[0],
                resolver=resolver,
            ),
            resolver=resolver,
        )
    if fullname in _GET_ITEM_FULLNAMES and len(arguments) == 2:  # noqa: PLR2004
        item = _selected_item(base=arguments[0], index=arguments[1])
        return (
            None
            if item is None
            else _expression_length(
                expression=item,
                fixed_tuple_lengths=fixed_tuple_lengths,
                resolver=resolver,
                class_info=class_info,
            )
        )
    return _fixed_tuple_annotation_length(
        annotation=_returned_annotation(node=resolver.node(name)),
        resolver=resolver,
    )


# Names which conventionally refer to the object a method is called on.
# Only an attribute of one of these is looked up on the checked class;
# any other name could refer to anything at all.
_SELF_NAMES = frozenset({"cls", "self"})


def _attribute_annotation(
    *,
    expression: MemberExpr,
    class_info: TypeInfo,
) -> Type | None:
    """Return the annotation of an attribute of the checked class."""
    base = expression.expr
    if not isinstance(base, NameExpr) or base.name not in _SELF_NAMES:
        return None
    symbol = class_info.get(name=expression.name)
    node = None if symbol is None else symbol.node
    return node.type if isinstance(node, Var) else None


def _resolved_expression_length(
    *,
    expression: Expression,
    fixed_tuple_lengths: dict[str, int],
    resolver: "_AnnotationResolver",
    class_info: TypeInfo,
) -> int | None:
    """Return the length an expression produces, using declarations.

    This resolves forms whose length comes from a declaration rather
    than from the expression itself, such as a call to an annotated
    function or an annotated attribute.
    """
    match expression:
        case AwaitExpr(expr=awaited):
            return _expression_length(
                expression=awaited,
                fixed_tuple_lengths=fixed_tuple_lengths,
                resolver=resolver,
                class_info=class_info,
            )
        case CallExpr() as call:
            return _called_expression_length(
                expression=call,
                fixed_tuple_lengths=fixed_tuple_lengths,
                resolver=resolver,
                class_info=class_info,
            )
        case MemberExpr() as member:
            return _fixed_tuple_annotation_length(
                annotation=_attribute_annotation(
                    expression=member,
                    class_info=class_info,
                ),
                resolver=resolver,
            )
        case _:
            return None


def _expression_length(
    *,
    expression: Expression,
    fixed_tuple_lengths: dict[str, int],
    resolver: "_AnnotationResolver",
    class_info: TypeInfo,
) -> int | None:
    """Return the length an expression produces, if it is known."""
    resolved = _resolved_expression_length(
        expression=expression,
        fixed_tuple_lengths=fixed_tuple_lengths,
        resolver=resolver,
        class_info=class_info,
    )
    if resolved is not None:
        return resolved
    return _known_sequence_length(
        expression=expression,
        fixed_tuple_lengths=fixed_tuple_lengths,
    )


def _spread_positional_counts(
    *,
    call: CallExpr,
    fixed_tuple_lengths: dict[str, int],
    resolver: "_AnnotationResolver",
    class_info: TypeInfo,
) -> dict[int, int]:
    """Return the positions each ``*`` argument is known to fill.

    These are worked out while the syntax tree is still available, and
    are used later wherever the call is checked.
    """
    counts: dict[int, int] = {}
    arguments = zip(call.args, call.arg_kinds, strict=True)
    for index, (argument, kind) in enumerate(iterable=arguments):
        if kind == ArgKind.ARG_STAR:
            counts[index] = _spread_positional_count(
                expression=argument,
                fixed_tuple_lengths=fixed_tuple_lengths,
                resolver=resolver,
                class_info=class_info,
            )
    return counts


def _spread_positional_count(
    *,
    expression: Expression,
    fixed_tuple_lengths: dict[str, int],
    resolver: "_AnnotationResolver",
    class_info: TypeInfo,
) -> int:
    """Return positions known to be filled by a ``*`` argument."""
    resolved_length = _resolved_expression_length(
        expression=expression,
        fixed_tuple_lengths=fixed_tuple_lengths,
        resolver=resolver,
        class_info=class_info,
    )
    if resolved_length is not None:
        return resolved_length
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


def _formals_disallow_positional(
    *,
    transformed: CallableType,
    first_formal_index: int,
    positional_argument_count: int,
) -> bool:
    """Return whether positions reach a keyword-only parameter."""
    formal_arg_index = first_formal_index
    for _ in range(positional_argument_count):
        if formal_arg_index >= len(transformed.arg_kinds):
            return False

        formal_arg_kind = transformed.arg_kinds[formal_arg_index]
        if formal_arg_kind == ArgKind.ARG_STAR:
            return False
        formal_arg_index += 1

        if formal_arg_kind in {ArgKind.ARG_NAMED, ArgKind.ARG_NAMED_OPT}:
            return True
    return False


def _call_disallows_positional_argument(
    *,
    call: CallExpr,
    signature: CallableType,
    fullname: str,
    ignore_names: list[str],
    skip_bound_argument: bool,
    spread_lengths: dict[int, int],
) -> bool:
    """Return whether a call passes a transformed argument by position."""
    transformed = _transform_callable_type(
        signature=signature,
        fullname=fullname,
        ignore_names=ignore_names,
        skip_bound_argument=skip_bound_argument,
        preserved_positional_argument_count=0,
    )

    positional_argument_counts: list[int] = []
    for index, actual_arg_kind in enumerate(iterable=call.arg_kinds):
        if actual_arg_kind == ArgKind.ARG_POS:
            positional_argument_counts.append(1)
        elif actual_arg_kind == ArgKind.ARG_STAR:
            positional_argument_counts.append(spread_lengths.get(index, 0))

    return _formals_disallow_positional(
        transformed=transformed,
        first_formal_index=1 if skip_bound_argument else 0,
        positional_argument_count=sum(positional_argument_counts),
    )


def _super_call_disallows_positional_argument(
    *,
    call: CallExpr,
    signature: FunctionLike,
    fullname: str,
    ignore_names: list[str],
    skip_bound_argument: bool,
    spread_lengths: dict[int, int],
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
            spread_lengths=spread_lengths,
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


# Generic types whose single type argument is the type of the values
# obtained by iterating over them.
_ITERABLE_FULLNAMES = frozenset(
    {
        "builtins.frozenset",
        "builtins.list",
        "builtins.set",
        "collections.abc.AsyncIterable",
        "collections.abc.AsyncIterator",
        "collections.abc.Collection",
        "collections.abc.Iterable",
        "collections.abc.Iterator",
        "collections.abc.Sequence",
        "typing.AsyncIterable",
        "typing.AsyncIterator",
        "typing.Collection",
        "typing.FrozenSet",
        "typing.Iterable",
        "typing.Iterator",
        "typing.List",
        "typing.Sequence",
        "typing.Set",
    }
)


def _element_annotation(
    *,
    annotation: Type,
    resolver: "_AnnotationResolver",
) -> Type | None:
    """Return the annotation of the values an iterable yields."""
    if not isinstance(annotation, UnboundType):
        return None
    fullname = resolver.fullname(annotation.name)
    if fullname in _ITERABLE_FULLNAMES and len(annotation.args) == 1:
        return annotation.args[0]
    if fullname in _TUPLE_FULLNAMES and len(annotation.args) == 2:  # noqa: PLR2004
        # ``tuple[X, ...]`` yields values of type ``X``.
        if isinstance(annotation.args[1], EllipsisType):
            return annotation.args[0]
        return None
    return None


def _iterated_element_length(
    *,
    iterable: Expression,
    calls: _CollectedCalls,
) -> int | None:
    """Return the fixed tuple length of the values an iterable yields."""
    if not isinstance(iterable, NameExpr):
        return None
    annotation = calls.annotation(iterable.name)
    if annotation is None:
        return None
    element = _element_annotation(
        annotation=annotation,
        resolver=calls.resolver,
    )
    if element is None:
        return None
    return _fixed_tuple_annotation_length(
        annotation=element,
        resolver=calls.resolver,
    )


def _bind_iteration_target(
    *,
    index: Expression,
    iterable: Expression,
    calls: _CollectedCalls,
) -> None:
    """Record the names a loop or comprehension target binds."""
    if not isinstance(index, NameExpr):
        calls.bind(_binding_target_names(index))
        return
    calls.bind_length(
        name=index.name,
        length=_iterated_element_length(iterable=iterable, calls=calls),
    )


def _bind_pattern_capture(
    *,
    pattern: Pattern,
    subject: Expression,
    calls: _CollectedCalls,
) -> None:
    """Record the names a match pattern captures."""
    match pattern:
        case AsPattern(pattern=None, name=NameExpr(name=name)):
            calls.bind_length(
                name=name,
                length=_known_sequence_length(
                    expression=subject,
                    fixed_tuple_lengths=calls.visible_lengths(),
                ),
            )
        case _:
            calls.bind(_pattern_bound_names(pattern))


_CONTEXT_MANAGER_FULLNAMES = frozenset(
    {"contextlib.asynccontextmanager", "contextlib.contextmanager"}
)


def _is_context_manager_factory(
    *,
    node: SymbolNode | None,
    resolver: "_AnnotationResolver",
) -> bool:
    """Return whether a definition is decorated as a context manager."""
    if not isinstance(node, Decorator):
        return False
    for decorator in node.decorators:
        name = _dotted_name(expression=decorator)
        if (
            name is not None
            and resolver.fullname(name) in _CONTEXT_MANAGER_FULLNAMES
        ):
            return True
    return False


def _context_manager_element_length(
    *,
    expression: Expression,
    calls: _CollectedCalls,
) -> int | None:
    """Return the fixed tuple length a context manager yields."""
    if not isinstance(expression, CallExpr):
        return None
    name = _dotted_name(expression=expression.callee)
    if name is None:
        return None
    node = calls.resolver.node(name)
    if not _is_context_manager_factory(node=node, resolver=calls.resolver):
        return None

    returned = _returned_annotation(node=node)
    if returned is None:
        return None
    element = _element_annotation(
        annotation=returned,
        resolver=calls.resolver,
    )
    if element is None:
        return None
    return _fixed_tuple_annotation_length(
        annotation=element,
        resolver=calls.resolver,
    )


def _bind_context_manager_target(
    *,
    target: Expression,
    expression: Expression,
    calls: _CollectedCalls,
) -> None:
    """Record the names a ``with`` statement target binds."""
    if not isinstance(target, NameExpr):
        calls.bind(_binding_target_names(target))
        return
    calls.bind_length(
        name=target.name,
        length=_context_manager_element_length(
            expression=expression,
            calls=calls,
        ),
    )


def _bind_assignment_target(
    *,
    lvalue: Expression,
    assignment: AssignmentStmt,
    calls: _CollectedCalls,
) -> None:
    """Record the names an assignment binds, and their lengths."""
    names = _binding_target_names(lvalue)
    if not isinstance(lvalue, NameExpr):
        calls.bind(names)
        return

    if assignment.unanalyzed_type is None:
        length = _known_sequence_length(
            expression=assignment.rvalue,
            fixed_tuple_lengths=calls.visible_lengths(),
        )
    else:
        # An explicit annotation describes the name everywhere, so a
        # variable-length annotation wins over the assigned value.
        length = _fixed_tuple_annotation_length(
            annotation=assignment.unanalyzed_type,
            resolver=calls.resolver,
        )
    calls.bind_length(name=lvalue.name, length=length)


def _collect_call_exprs_from_statement(  # noqa: C901, PLR0912, PLR0915  # pylint: disable=too-complex,too-many-branches,too-many-statements
    statement: Statement,
    calls: _CollectedCalls,
    /,
) -> None:
    """Collect call expressions from a statement."""
    match statement:
        case ExpressionStmt(expr=expr):
            _collect_call_exprs(expr, calls)
        case AssignmentStmt(rvalue=rvalue, lvalues=lvalues) as assignment:
            _collect_call_exprs(rvalue, calls)
            for lvalue in lvalues:
                _bind_assignment_target(
                    lvalue=lvalue,
                    assignment=assignment,
                    calls=calls,
                )
                _collect_call_exprs(lvalue, calls)
        case OperatorAssignmentStmt(rvalue=rvalue, lvalue=lvalue):
            _collect_call_exprs(rvalue, calls)
            calls.bind(_binding_target_names(lvalue))
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
            _bind_iteration_target(index=index, iterable=expr, calls=calls)
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
            calls.bind(_binding_target_names(expr))
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
                    calls.bind(_binding_target_names(variable))
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
                    _bind_context_manager_target(
                        target=target,
                        expression=expression,
                        calls=calls,
                    )
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
                _bind_pattern_capture(
                    pattern=pattern,
                    subject=subject,
                    calls=calls,
                )
                _collect_call_exprs_from_pattern(pattern, calls)
                if guard is not None:
                    _collect_call_exprs(guard, calls)
                _collect_call_exprs(body, calls)
        case Block(body=body):
            for body_statement in body:
                _collect_call_exprs(body_statement, calls)
        case FuncDef(name=name):
            calls.bind({name})
            _collect_call_exprs_from_func_item(statement, calls)
        case OverloadedFuncDef(items=items, name=name):
            calls.bind({name})
            for overload_item in items:
                _collect_call_exprs(overload_item, calls)
        case Decorator(func=func, decorators=decorators, name=name):
            calls.bind({name})
            _collect_call_exprs(func, calls)
            for decorator in decorators:
                _collect_call_exprs(decorator, calls)
        case ClassDef(
            decorators=decorators,
            base_type_exprs=base_type_exprs,
            metaclass=metaclass,
            keywords=keywords,
            name=name,
        ):
            calls.bind({name})
            for decorator in decorators:
                _collect_call_exprs(decorator, calls)
            for base_type_expression in base_type_exprs:
                _collect_call_exprs(base_type_expression, calls)
            if metaclass is not None:
                _collect_call_exprs(metaclass, calls)
            for keyword_expression in keywords.values():
                _collect_call_exprs(keyword_expression, calls)
        case GlobalDecl(names=names):
            calls.bind(set(names))
        case NonlocalDecl(names=names):
            calls.declare_nonlocal(set(names))
        case Import(ids=ids):
            calls.bind(
                {
                    as_name or identifier.split(sep=".", maxsplit=1)[0]
                    for identifier, as_name in ids
                }
            )
        case ImportFrom(names=imported_names):
            calls.bind(
                {
                    as_name or imported_name
                    for imported_name, as_name in imported_names
                }
            )
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
    parameter_annotations = {
        argument.variable.name: argument.type_annotation
        for argument in func_item.arguments
        if argument.type_annotation is not None
    }
    calls.push_scope(
        is_comprehension=False,
        annotations=parameter_annotations,
    )
    _collect_call_exprs(func_item.body, calls)
    scope = calls.pop_scope()
    rebound_names = scope.names

    parameter_names = {
        argument.variable.name for argument in func_item.arguments
    }
    fixed_tuple_lengths = {
        argument.variable.name: fixed_tuple_length
        for argument in func_item.arguments
        # A parameter which the body binds again no longer holds the
        # annotated tuple everywhere in the scope.
        if argument.variable.name not in rebound_names
        and (
            fixed_tuple_length := _fixed_tuple_annotation_length(
                annotation=argument.type_annotation,
                resolver=calls.resolver,
            )
        )
        is not None
    }
    fixed_tuple_lengths.update(
        {
            name: length
            for name, length in scope.assigned_lengths.items()
            if length is not None
        }
    )
    _apply_scope_to_calls(
        calls=calls,
        first_call_index=first_body_call_index,
        names=parameter_names | rebound_names,
        fixed_tuple_lengths=fixed_tuple_lengths,
    )


# Names which an annotation may resolve to, looked up through the
# semantic analyzer so that import aliases such as ``from typing import
# Tuple as FixedTuple`` are recognized.
_TUPLE_FULLNAMES = frozenset({"builtins.tuple", "typing.Tuple"})
_ANNOTATED_FULLNAMES = frozenset(
    {"typing.Annotated", "typing_extensions.Annotated"}
)
_UNPACK_FULLNAMES = frozenset({"typing.Unpack", "typing_extensions.Unpack"})
_NEVER_FULLNAMES = frozenset(
    {
        "typing.NoReturn",
        "typing.Never",
        "typing_extensions.NoReturn",
        "typing_extensions.Never",
    }
)


@dataclass(frozen=True, kw_only=True)
class _AnnotationResolver:
    """Resolves the names used in an annotation before analysis."""

    api: SemanticAnalyzerPluginInterface
    context: Context

    def fullname(self, name: str, /) -> str | None:
        """Return the full name an annotation name refers to."""
        symbol = self.api.lookup_qualified(
            name=name,
            ctx=self.context,
            suppress_errors=True,
        )
        return None if symbol is None else symbol.fullname

    def node(self, name: str, /) -> SymbolNode | None:
        """Return the symbol an annotation name refers to."""
        symbol = self.api.lookup_qualified(
            name=name,
            ctx=self.context,
            suppress_errors=True,
        )
        return None if symbol is None else symbol.node


def _fixed_tuple_type_length(
    *,
    tuple_type: Type | None,
    type_var_tuple_length: int | None,
) -> int | None:
    """Return the length of an analyzed tuple type, if known.

    ``type_var_tuple_length`` is the number of items a ``TypeVarTuple``
    stands for, or ``None`` when that is not known.
    """
    proper_type = get_proper_type(typ=tuple_type)
    if isinstance(proper_type, Instance):
        proper_type = proper_type.type.tuple_type
    if not isinstance(proper_type, TupleType):
        return None

    length = 0
    for item in proper_type.items:
        proper_item = get_proper_type(typ=item)
        if not isinstance(proper_item, UnpackType):
            length += 1
        elif isinstance(
            get_proper_type(typ=proper_item.type), TypeVarTupleType
        ):
            if type_var_tuple_length is None:
                return None
            length += type_var_tuple_length
        else:
            item_length = _fixed_tuple_type_length(
                tuple_type=proper_item.type,
                type_var_tuple_length=None,
            )
            if item_length is None:
                return None
            length += item_length
    return length


def _alias_fixed_tuple_length(
    *,
    alias: TypeAlias,
    annotation: UnboundType,
) -> int | None:
    """Return the length of a tuple type alias, if known."""
    type_var_tuple_count = len(
        [
            type_var
            for type_var in alias.alias_tvars
            if isinstance(type_var, TypeVarTupleType)
        ]
    )
    type_var_tuple_length = None
    if type_var_tuple_count == 1:
        # Every argument which is not taken by another type variable is
        # taken by the single ``TypeVarTuple``.
        type_var_tuple_length = max(
            len(annotation.args) - len(alias.alias_tvars) + 1,
            0,
        )
    return _fixed_tuple_type_length(
        tuple_type=alias.target,
        type_var_tuple_length=type_var_tuple_length,
    )


def _unpacked_annotation(
    *,
    annotation: Type,
    resolver: _AnnotationResolver,
) -> Type | None:
    """Return the annotation unpacked by a PEP 646 ``*`` item.

    ``None`` means that the item is not an unpack.  Both the star syntax
    (``*tuple[int, int]``) and the ``Unpack[...]`` spelling are
    recognized.
    """
    if isinstance(annotation, UnpackType):
        return annotation.type
    if isinstance(annotation, UnboundType) and len(annotation.args) == 1:
        if resolver.fullname(annotation.name) in _UNPACK_FULLNAMES:
            return annotation.args[0]
        return None
    return None


def _literal_tuple_annotation_length(
    *,
    annotation: UnboundType,
    resolver: _AnnotationResolver,
) -> int | None:
    """Return the length of a written-out tuple annotation, if known."""
    if annotation.empty_tuple_index:
        return 0
    if not annotation.args or isinstance(annotation.args[-1], EllipsisType):
        return None

    length = 0
    for item in annotation.args:
        unpacked = _unpacked_annotation(annotation=item, resolver=resolver)
        if unpacked is None:
            length += 1
            continue
        unpacked_length = _fixed_tuple_annotation_length(
            annotation=unpacked,
            resolver=resolver,
        )
        if unpacked_length is None:
            return None
        length += unpacked_length
    return length


def _unbound_fixed_tuple_length(
    *,
    annotation: UnboundType,
    resolver: _AnnotationResolver,
) -> int | None:
    """Return the length of a named annotation, if known."""
    fullname = resolver.fullname(annotation.name)
    if fullname in _ANNOTATED_FULLNAMES:
        return _fixed_tuple_annotation_length(
            annotation=next(iter(annotation.args), None),
            resolver=resolver,
        )
    if fullname in _TUPLE_FULLNAMES:
        return _literal_tuple_annotation_length(
            annotation=annotation,
            resolver=resolver,
        )

    node = resolver.node(annotation.name)
    if isinstance(node, TypeAlias):
        return _alias_fixed_tuple_length(alias=node, annotation=annotation)
    if isinstance(node, TypeInfo):
        # A ``NamedTuple`` or a ``NewType`` over a fixed tuple.
        return _fixed_tuple_type_length(
            tuple_type=node.tuple_type,
            type_var_tuple_length=None,
        )
    return None


def _union_fixed_tuple_length(
    *,
    items: list[Type],
    resolver: _AnnotationResolver,
) -> int | None:
    """Return the length shared by every inhabited union item."""
    lengths: set[int] = set()
    for item in items:
        if (
            isinstance(item, UnboundType)
            and resolver.fullname(item.name) in _NEVER_FULLNAMES
        ):
            continue
        length = _fixed_tuple_annotation_length(
            annotation=item,
            resolver=resolver,
        )
        if length is None:
            return None
        lengths.add(length)
    if len(lengths) != 1:
        return None
    return lengths.pop()


def _fixed_tuple_annotation_length(
    *,
    annotation: Type | None,
    resolver: _AnnotationResolver,
) -> int | None:
    """Return the length of a fixed tuple annotation, if known."""
    match annotation:
        case UnionType(items=items):
            return _union_fixed_tuple_length(items=items, resolver=resolver)
        case UnboundType():
            return _unbound_fixed_tuple_length(
                annotation=annotation,
                resolver=resolver,
            )
        case _:
            return _fixed_tuple_type_length(
                tuple_type=annotation,
                type_var_tuple_length=None,
            )


def _collect_call_exprs_from_comprehension(
    *,
    indices: list[Expression],
    sequences: list[Expression],
    condlists: list[list[Expression]],
    results: list[Expression],
    calls: _CollectedCalls,
) -> None:
    """Collect call expressions from a comprehension.

    A comprehension is its own scope, so its targets shadow bindings of
    the same name in enclosing scopes.
    """
    # The leftmost iterable is evaluated in the enclosing scope, so the
    # comprehension's targets do not shadow anything within it.
    _collect_call_exprs(sequences[0], calls)
    first_call_index = len(calls)
    calls.push_scope(is_comprehension=True, annotations={})
    for position, (index, sequence, conditions) in enumerate(
        iterable=zip(indices, sequences, condlists, strict=True)
    ):
        if position:
            _collect_call_exprs(sequence, calls)
        _bind_iteration_target(index=index, iterable=sequence, calls=calls)
        _collect_call_exprs(index, calls)
        for condition in conditions:
            _collect_call_exprs(condition, calls)
    for result in results:
        _collect_call_exprs(result, calls)
    scope = calls.pop_scope()
    _apply_scope_to_calls(
        calls=calls,
        first_call_index=first_call_index,
        names=scope.names,
        fixed_tuple_lengths={
            name: length
            for name, length in scope.assigned_lengths.items()
            if length is not None
        },
    )


def _collect_call_exprs_from_expression(  # noqa: C901, PLR0912, PLR0915  # pylint: disable=too-complex,too-many-branches
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
            calls.bind_in_function_scope(_binding_target_names(target))
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
            _collect_call_exprs_from_comprehension(
                indices=indices,
                sequences=sequences,
                condlists=condlists,
                results=[left_expr],
                calls=calls,
            )
        case DictionaryComprehension(
            indices=indices,
            sequences=sequences,
            condlists=condlists,
            key=key,
            value=value,
        ):
            _collect_call_exprs_from_comprehension(
                indices=indices,
                sequences=sequences,
                condlists=condlists,
                results=[key, value],
                calls=calls,
            )
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


def _iter_call_exprs(
    node: _CallExprContainer,
    /,
    *,
    resolver: _AnnotationResolver,
) -> _CollectedCalls:
    """Return call expressions contained in a node."""
    calls = _CollectedCalls(resolver=resolver)
    _collect_call_exprs(node, calls)
    return calls


@dataclass(frozen=True, kw_only=True)
class _PendingSuperCall:
    """A ``super()`` call whose member type is not known yet.

    A member which is not a method -- an assigned function, a callable
    object, a descriptor -- has no type during semantic analysis, so the
    call is checked later from the attribute hook.
    """

    call: CallExpr
    method_name: str
    class_name: str
    spread_lengths: dict[int, int]


_PendingSuperCalls = dict[tuple[str, int, int], _PendingSuperCall]


def _pending_super_call_key(
    *,
    path: str,
    expr: CallExpr,
) -> tuple[str, int, int]:
    """Return the key identifying a ``super()`` call expression."""
    callee = expr.callee
    return (path, callee.line, callee.column)


def _instance_call_signature(*, instance: Instance) -> FunctionLike | None:
    """Return the signature of a callable object's ``__call__``."""
    symbol = instance.type.get(name="__call__")
    signature = None if symbol is None else get_proper_type(typ=symbol.type)
    return signature if isinstance(signature, FunctionLike) else None


def _check_pending_super_call(
    ctx: AttributeContext,
    *,
    fullname: str,
    ignore_names: list[str],
    pending_super_calls: _PendingSuperCalls,
    debug: bool,
) -> Type:
    """Check a ``super()`` call to a member which is not a method."""
    context = ctx.context
    if not isinstance(context, SuperExpr):
        return ctx.default_attr_type

    pending = pending_super_calls.get(
        (ctx.api.path, context.line, context.column)
    )
    if pending is None:
        return ctx.default_attr_type

    if debug:
        _write_debug_fullname(fullname=fullname, path=ctx.api.path)

    if fullname in ignore_names:
        return ctx.default_attr_type

    attribute_type = get_proper_type(typ=ctx.default_attr_type)
    signature: FunctionLike | None = None
    skip_bound_argument = False
    if isinstance(attribute_type, FunctionLike):
        signature = attribute_type
    elif isinstance(attribute_type, Instance):
        # A callable object supplies its own ``self``.
        signature = _instance_call_signature(instance=attribute_type)
        skip_bound_argument = True

    if signature is not None and _super_call_disallows_positional_argument(
        call=pending.call,
        signature=signature,
        fullname=fullname,
        ignore_names=ignore_names,
        skip_bound_argument=skip_bound_argument,
        spread_lengths=pending.spread_lengths,
    ):
        ctx.api.fail(
            f'Too many positional arguments for "{pending.method_name}" '
            f'of "{pending.class_name}"',
            pending.call,
            code=CALL_ARG,
        )
    return ctx.default_attr_type


def _check_super_method_call(
    *,
    ctx: ClassDefContext,
    expr: CallExpr,
    method_name: str,
    ignore_names: list[str],
    fixed_tuple_lengths: dict[str, int],
    resolver: "_AnnotationResolver",
    pending_super_calls: _PendingSuperCalls,
    path: str,
    debug: bool,
) -> None:
    """Check one ``super()`` method call expression."""
    for info in _super_method_mro(ctx=ctx, expr=expr):
        spread_lengths = _spread_positional_counts(
            call=expr,
            fixed_tuple_lengths=fixed_tuple_lengths,
            resolver=resolver,
            class_info=ctx.cls.info,
        )
        symbol = info.names.get(method_name)
        if symbol is None:
            continue

        match symbol.node:
            case FuncDef() | OverloadedFuncDef() as node:
                fullname = node.fullname
                typ = node.type
                skip_bound_argument = node.has_self_or_cls_argument
            case Decorator() as node:
                fullname = node.fullname
                typ = node.func.type
                skip_bound_argument = node.func.has_self_or_cls_argument
            case _:
                # The member is not a method, so its type is only known
                # once type checking runs.
                pending_super_calls[
                    _pending_super_call_key(path=path, expr=expr)
                ] = _PendingSuperCall(
                    call=expr,
                    method_name=method_name,
                    class_name=info.name,
                    spread_lengths=spread_lengths,
                )
                return

        if debug:
            _write_debug_fullname(fullname=fullname, path=path)

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
            spread_lengths=spread_lengths,
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


def _module_assigned_lengths(
    *,
    module: MypyFile,
    resolver: _AnnotationResolver,
) -> dict[str, int]:
    """Return module-level names assigned a sequence of known length."""
    scope_calls = _CollectedCalls(resolver=resolver)
    for statement in module.defs:
        _collect_call_exprs(statement, scope_calls)
    return {
        name: length
        for name, length in scope_calls.pop_scope().assigned_lengths.items()
        if length is not None
    }


def _check_super_method_calls(
    ctx: ClassDefContext,
    *,
    ignore_names: list[str],
    module_lengths: dict[str, dict[str, int]],
    pending_super_calls: _PendingSuperCalls,
    debug: bool,
) -> None:
    """Check ``super()`` method calls in a class body.

    Needed because ``mypy`` does not run method signature hooks for
    ``super().method(...)``
    (https://github.com/python/mypy/issues/21744).
    """
    module = ctx.api.modules[ctx.api.cur_mod_id]
    path = module.path
    resolver = _AnnotationResolver(api=ctx.api, context=ctx.cls)
    calls = _iter_call_exprs(ctx.cls.defs, resolver=resolver)
    if ctx.api.cur_mod_id not in module_lengths:
        module_lengths[ctx.api.cur_mod_id] = _module_assigned_lengths(
            module=module,
            resolver=resolver,
        )
    _apply_scope_to_calls(
        calls=calls,
        first_call_index=0,
        names=set(module_lengths[ctx.api.cur_mod_id]),
        fixed_tuple_lengths=module_lengths[ctx.api.cur_mod_id],
    )
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
            resolver=resolver,
            pending_super_calls=pending_super_calls,
            path=path,
            debug=debug,
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
        self._pending_super_calls: _PendingSuperCalls = {}
        self._module_lengths: dict[str, dict[str, int]] = {}
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
            module_lengths=self._module_lengths,
            pending_super_calls=self._pending_super_calls,
            debug=self._debug,
        )

    def get_attribute_hook(
        self,
        fullname: str,
    ) -> Callable[[AttributeContext], Type] | None:
        """Check ``super()`` calls to members which are not methods.

        The type of such a member is only known once type checking
        runs, which is after the base class hook which finds the call.
        """
        return partial(
            _check_pending_super_call,
            fullname=fullname,
            ignore_names=self._ignore_names,
            pending_super_calls=self._pending_super_calls,
            debug=self._debug,
        )


def plugin(version: str) -> type[KeywordOnlyPlugin]:
    """Plugin entry point."""
    del version  # to satisfy vulture
    return KeywordOnlyPlugin
