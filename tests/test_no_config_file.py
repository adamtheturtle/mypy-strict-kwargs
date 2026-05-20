"""Test plugin behavior when no configuration file is provided."""

# pylint: disable=import-private-name,protected-access,unsupported-assignment-operation,wrong-spelling-in-comment
# pyright: reportPrivateUsage=false

from typing import cast

import pytest
from mypy.errorcodes import MISC
from mypy.nodes import (
    ARG_POS,
    MDEF,
    ArgKind,
    Argument,
    Block,
    CallExpr,
    ClassDef,
    Decorator,
    FuncDef,
    MemberExpr,
    NameExpr,
    SuperExpr,
    SymbolTable,
    SymbolTableNode,
    TypeInfo,
    Var,
)
from mypy.options import Options
from mypy.plugin import ClassDefContext, SemanticAnalyzerPluginInterface
from mypy.test.typefixture import TypeFixture
from mypy.types import (
    AnyType,
    CallableType,
    FunctionLike,
    Overloaded,
    TypeOfAny,
)

from mypy_strict_kwargs.plugin import (
    KeywordOnlyPlugin,
    _call_disallows_positional_argument,
    _callable_items,
    _check_super_method_call,
    _is_super_expr,
    _iter_call_exprs,
    _iter_child_nodes,
    _super_call_disallows_positional_argument,
    _super_method_name,
)

_T = TypeFixture()


def test_no_config_file() -> None:
    """
    Test that plugin provides hooks when no configuration file
    exists.
    """
    options = Options()

    plugin = KeywordOnlyPlugin(options=options)

    assert plugin.get_base_class_hook(fullname="test.Base") is not None
    assert plugin.get_function_signature_hook(fullname="test.func") is not None
    assert plugin.get_method_signature_hook(fullname="test.method") is not None


def _signature(
    *,
    arg_kinds: list[ArgKind],
    arg_names: list[str | None],
) -> CallableType:
    """Return a callable type with unimportant argument and return
    types.
    """
    any_type = AnyType(type_of_any=TypeOfAny.explicit)
    return CallableType(
        arg_types=[any_type] * len(arg_kinds),
        arg_kinds=arg_kinds,
        arg_names=arg_names,
        ret_type=any_type,
        fallback=_T.function,
    )


def _call(
    *,
    callee: NameExpr | MemberExpr | SuperExpr,
    arg_kinds: list[ArgKind],
) -> CallExpr:
    """Return a call expression with placeholder arguments."""
    return CallExpr(
        callee=callee,
        args=[NameExpr(name="x") for _ in arg_kinds],
        arg_kinds=arg_kinds,
        arg_names=[None] * len(arg_kinds),
    )


def test_callable_items() -> None:
    """Test unpacking callable and overloaded signatures."""
    signature = _signature(arg_kinds=[], arg_names=[])
    overloaded = Overloaded(items=[signature])

    assert _callable_items(signature=signature) == [signature]
    assert _callable_items(signature=overloaded) == [signature]

    with pytest.raises(expected_exception=TypeError):
        _callable_items(signature=cast("FunctionLike", object()))


def test_super_expression_detection() -> None:
    """Test detection of super expressions and super method names."""
    super_name = NameExpr(name="super")
    super_fullname = NameExpr(name="not_super")
    super_fullname.fullname = "builtins.super"
    regular_name = NameExpr(name="not_super")

    super_call = _call(callee=super_name, arg_kinds=[])
    fullname_super_call = _call(callee=super_fullname, arg_kinds=[])
    regular_call = _call(callee=regular_name, arg_kinds=[])

    assert _is_super_expr(expr=super_call)
    assert _is_super_expr(
        expr=SuperExpr(name="method", call=super_call),
    )
    assert _is_super_expr(expr=fullname_super_call)
    assert not _is_super_expr(expr=regular_call)

    assert (
        _super_method_name(
            expr=_call(
                callee=SuperExpr(name="method", call=super_call),
                arg_kinds=[],
            ),
        )
        == "method"
    )
    assert (
        _super_method_name(
            expr=_call(
                callee=MemberExpr(expr=super_call, name="method"),
                arg_kinds=[],
            ),
        )
        == "method"
    )
    assert _super_method_name(expr=regular_call) is None
    assert (
        _super_method_name(
            expr=_call(
                callee=MemberExpr(expr=NameExpr(name="obj"), name="method"),
                arg_kinds=[],
            ),
        )
        is None
    )
    assert (
        _super_method_name(
            expr=_call(
                callee=MemberExpr(expr=regular_call, name="method"),
                arg_kinds=[],
            ),
        )
        is None
    )


def test_call_disallows_positional_argument() -> None:
    """Test positional argument rejection for transformed call
    signatures.
    """
    signature = _signature(
        arg_kinds=[ARG_POS, ARG_POS],
        arg_names=["self", "a"],
    )

    assert _call_disallows_positional_argument(
        call=_call(callee=NameExpr(name="method"), arg_kinds=[ARG_POS]),
        signature=signature,
        fullname="main.B.method",
        ignore_names=[],
        skip_bound_argument=True,
    )
    assert not _call_disallows_positional_argument(
        call=_call(callee=NameExpr(name="method"), arg_kinds=[]),
        signature=signature,
        fullname="main.B.method",
        ignore_names=[],
        skip_bound_argument=True,
    )
    assert not _call_disallows_positional_argument(
        call=_call(
            callee=NameExpr(name="method"),
            arg_kinds=[ArgKind.ARG_NAMED],
        ),
        signature=signature,
        fullname="main.B.method",
        ignore_names=[],
        skip_bound_argument=True,
    )
    ignored_signature = _signature(arg_kinds=[ARG_POS], arg_names=["a"])
    assert not _call_disallows_positional_argument(
        call=_call(callee=NameExpr(name="func"), arg_kinds=[ARG_POS]),
        signature=ignored_signature,
        fullname="main.func",
        ignore_names=["main.func"],
        skip_bound_argument=False,
    )
    assert not _call_disallows_positional_argument(
        call=_call(callee=NameExpr(name="func"), arg_kinds=[ARG_POS]),
        signature=_signature(arg_kinds=[], arg_names=[]),
        fullname="main.func",
        ignore_names=[],
        skip_bound_argument=False,
    )


def test_super_call_disallows_positional_argument() -> None:
    """Test overloaded super-call rejection only when all overloads reject."""
    rejecting_signature = _signature(arg_kinds=[ARG_POS], arg_names=["a"])
    accepting_signature = _signature(arg_kinds=[ARG_POS], arg_names=[None])
    call = _call(callee=NameExpr(name="method"), arg_kinds=[ARG_POS])

    assert _super_call_disallows_positional_argument(
        call=call,
        signature=Overloaded(items=[rejecting_signature]),
        fullname="main.B.method",
        ignore_names=[],
        skip_bound_argument=False,
    )
    assert not _super_call_disallows_positional_argument(
        call=call,
        signature=Overloaded(items=[rejecting_signature, accepting_signature]),
        fullname="main.B.method",
        ignore_names=[],
        skip_bound_argument=False,
    )


def test_iter_child_nodes() -> None:
    """Test child-node iteration across direct and repeated child
    nodes.
    """
    child = NameExpr(name="x")
    call = CallExpr(
        callee=child,
        args=[child],
        arg_kinds=[ARG_POS],
        arg_names=[None],
    )

    assert child in _iter_child_nodes(node=call)
    assert _iter_call_exprs(node=call) == [call]


class _Api:
    """Record plugin failures."""

    def __init__(self) -> None:
        """Initialize recorded failures."""
        self.failures: list[tuple[str, object, object]] = []

    def fail(self, *, msg: str, ctx: object, code: object = None) -> None:
        """Record a failure."""
        self.failures.append((msg, ctx, code))


def _type_info(*, name: str) -> TypeInfo:
    """Return a minimal type info."""
    class_def = ClassDef(name=name, defs=Block(body=[]))
    type_info = TypeInfo(
        names=SymbolTable(),
        defn=class_def,
        module_name="main",
    )
    class_def.info = type_info
    return type_info


def _method_node(
    *,
    typ: FunctionLike | None,
    fullname: str = "main.B.method",
) -> FuncDef:
    """Return a minimal method node."""
    method = FuncDef(
        name="method",
        arguments=[
            Argument(
                variable=Var(name="self"),
                type_annotation=None,
                initializer=None,
                kind=ARG_POS,
            ),
            Argument(
                variable=Var(name="a"),
                type_annotation=None,
                initializer=None,
                kind=ARG_POS,
            ),
        ],
        body=Block(body=[]),
        typ=typ,
    )
    method._fullname = fullname  # noqa: SLF001
    assert method._fullname == fullname  # noqa: SLF001
    return method


def _class_context(
    *,
    base_symbol: SymbolTableNode | None,
    other_base_symbol: SymbolTableNode | None = None,
) -> tuple[ClassDefContext, _Api]:
    """Return a class context with an optional base method symbol."""
    base_info = _type_info(name="B")
    if base_symbol is not None:
        base_info.names["method"] = base_symbol
    base_infos = [base_info]

    if other_base_symbol is not None:
        other_base_info = _type_info(name="A")
        other_base_info.names["method"] = other_base_symbol
        base_infos.append(other_base_info)

    class_info = _type_info(name="C")
    class_info.mro = [class_info, *base_infos]
    api = _Api()
    return (
        ClassDefContext(
            cls=class_info.defn,
            reason=NameExpr(name="B"),
            api=cast(SemanticAnalyzerPluginInterface, api),  # noqa: TC006
        ),
        api,
    )


def test_check_super_method_call() -> None:
    """Test reporting for super method calls."""
    signature = _signature(
        arg_kinds=[ARG_POS, ARG_POS],
        arg_names=["self", "a"],
    )
    call = _call(callee=NameExpr(name="method"), arg_kinds=[ARG_POS])

    ctx, api = _class_context(
        base_symbol=SymbolTableNode(
            kind=MDEF,
            node=_method_node(typ=signature),
        ),
    )

    _check_super_method_call(
        ctx=ctx,
        expr=call,
        method_name="method",
        ignore_names=[],
    )

    assert api.failures == [
        ('Too many positional arguments for "method" of "B"', call, MISC),
    ]


def test_check_super_method_call_ignored() -> None:
    """Test ignored super method calls."""
    signature = _signature(
        arg_kinds=[ARG_POS, ARG_POS],
        arg_names=["self", "a"],
    )
    ctx, api = _class_context(
        base_symbol=SymbolTableNode(
            kind=MDEF,
            node=_method_node(typ=signature),
        ),
    )

    _check_super_method_call(
        ctx=ctx,
        expr=_call(callee=NameExpr(name="method"), arg_kinds=[ARG_POS]),
        method_name="method",
        ignore_names=["main.B.method"],
    )

    assert not api.failures


def test_check_super_method_call_continues_past_accepting_base() -> None:
    """Test super method lookup continues after an accepting base
    method.
    """
    accepting_signature = _signature(
        arg_kinds=[ARG_POS, ARG_POS],
        arg_names=["self", None],
    )
    rejecting_signature = _signature(
        arg_kinds=[ARG_POS, ARG_POS],
        arg_names=["self", "a"],
    )
    call = _call(callee=NameExpr(name="method"), arg_kinds=[ARG_POS])
    ctx, api = _class_context(
        base_symbol=SymbolTableNode(
            kind=MDEF,
            node=_method_node(typ=accepting_signature),
        ),
        other_base_symbol=SymbolTableNode(
            kind=MDEF,
            node=_method_node(
                typ=rejecting_signature,
                fullname="main.A.method",
            ),
        ),
    )

    _check_super_method_call(
        ctx=ctx,
        expr=call,
        method_name="method",
        ignore_names=[],
    )

    assert api.failures == [
        ('Too many positional arguments for "method" of "A"', call, MISC),
    ]


def test_check_super_method_call_missing_or_non_function() -> None:
    """Test super method calls with no callable base method."""
    ctx, api = _class_context(base_symbol=None)
    _check_super_method_call(
        ctx=ctx,
        expr=_call(callee=NameExpr(name="method"), arg_kinds=[ARG_POS]),
        method_name="method",
        ignore_names=[],
    )
    assert not api.failures

    ctx, api = _class_context(
        base_symbol=SymbolTableNode(
            kind=MDEF,
            node=_method_node(typ=None),
        ),
    )
    _check_super_method_call(
        ctx=ctx,
        expr=_call(callee=NameExpr(name="method"), arg_kinds=[ARG_POS]),
        method_name="method",
        ignore_names=[],
    )
    assert not api.failures


def test_check_super_method_call_decorator_and_other_nodes() -> None:
    """Test decorated and unsupported base method symbols."""
    signature = _signature(
        arg_kinds=[ARG_POS, ARG_POS],
        arg_names=["self", "a"],
    )
    method = _method_node(typ=signature)
    decorated_method = Decorator(
        func=method,
        decorators=[],
        var=Var(name="method"),
    )

    ctx, api = _class_context(
        base_symbol=SymbolTableNode(kind=MDEF, node=decorated_method),
    )
    _check_super_method_call(
        ctx=ctx,
        expr=_call(callee=NameExpr(name="method"), arg_kinds=[ARG_POS]),
        method_name="method",
        ignore_names=[],
    )
    assert api.failures

    ctx, api = _class_context(
        base_symbol=SymbolTableNode(kind=MDEF, node=Var(name="method")),
    )
    _check_super_method_call(
        ctx=ctx,
        expr=_call(callee=NameExpr(name="method"), arg_kinds=[ARG_POS]),
        method_name="method",
        ignore_names=[],
    )
    assert not api.failures
