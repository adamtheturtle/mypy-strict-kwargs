"""Tests for collecting call expressions from mypy AST nodes."""

from mypy.nodes import (
    REVEAL_LOCALS,
    REVEAL_TYPE,
    ArgKind,
    AssertTypeExpr,
    CallExpr,
    CastExpr,
    IndexExpr,
    NameExpr,
    RevealExpr,
    StrExpr,
    TemplateStrExpr,
    TypeApplication,
)
from mypy.patterns import (
    AsPattern,
    ClassPattern,
    MappingPattern,
    OrPattern,
    SequencePattern,
    SingletonPattern,
    StarredPattern,
    ValuePattern,
)
from mypy.types import NoneType

from mypy_strict_kwargs.plugin import _collect_call_exprs, _iter_call_exprs


def _call_expr(name: str, /) -> CallExpr:
    """Build a simple call expression for tests."""
    return CallExpr(
        callee=NameExpr(name),
        args=[],
        arg_kinds=[ArgKind.ARG_POS],
        arg_names=[None],
    )


def test_ignores_unrecognized_nodes() -> None:
    """Nodes that are not calls, statements, expressions, or patterns are
    skipped.
    """
    calls: list[CallExpr] = []
    _collect_call_exprs(object(), calls)
    assert calls == []


def test_collects_calls_from_call_expr_analyzed() -> None:
    """``CallExpr.analyzed`` can contain nested call expressions."""
    inner_call = _call_expr("analyzed")
    outer_call = _call_expr("outer")
    outer_call.analyzed = inner_call
    calls: list[CallExpr] = []
    _collect_call_exprs(outer_call, calls)
    assert outer_call in calls
    assert inner_call in calls


def test_collects_calls_from_cast_expr() -> None:
    """``CastExpr`` wraps nested call expressions."""
    inner_call = _call_expr("super")
    calls: list[CallExpr] = []
    _collect_call_exprs(CastExpr(inner_call, NoneType()), calls)
    assert inner_call in calls


def test_collects_calls_from_assert_type_expr() -> None:
    """``AssertTypeExpr`` wraps nested call expressions."""
    inner_call = _call_expr("super")
    calls: list[CallExpr] = []
    _collect_call_exprs(AssertTypeExpr(inner_call, NoneType()), calls)
    assert inner_call in calls


def test_collects_calls_from_reveal_expr() -> None:
    """``RevealExpr`` wraps nested call expressions for
    ``reveal_type``.
    """
    inner_call = _call_expr("super")
    calls: list[CallExpr] = []
    _collect_call_exprs(RevealExpr(REVEAL_TYPE, inner_call), calls)
    _collect_call_exprs(RevealExpr(REVEAL_LOCALS), calls)
    assert inner_call in calls


def test_collects_calls_from_template_str_expr() -> None:
    """``TemplateStrExpr`` wraps interpolated call expressions."""
    inner_call = _call_expr("one")
    format_call = _call_expr("format")
    calls: list[CallExpr] = []
    _collect_call_exprs(
        TemplateStrExpr(
            items=[
                (inner_call, "x", None, None),
                (inner_call, "y", None, format_call),
                StrExpr(value="plain"),
            ],
        ),
        calls,
    )
    assert inner_call in calls
    assert format_call in calls


def test_collects_calls_from_index_expr_analyzed() -> None:
    """``IndexExpr.analyzed`` can contain nested call expressions."""
    inner_call = _call_expr("items")
    index_expr = IndexExpr(NameExpr("list"), NameExpr("int"))
    index_expr.analyzed = TypeApplication(inner_call, [])
    calls: list[CallExpr] = []
    _collect_call_exprs(index_expr, calls)
    assert inner_call in calls


def test_collects_calls_from_type_application() -> None:
    """``TypeApplication`` wraps call expressions in its base."""
    inner_call = _call_expr("items")
    calls: list[CallExpr] = []
    _collect_call_exprs(TypeApplication(inner_call, []), calls)
    assert inner_call in calls


def test_collects_calls_from_match_patterns() -> None:
    """Match patterns can contain nested call expressions."""
    value_call = _call_expr("one")
    other_call = _call_expr("two")
    mapping_key_call = _call_expr("key")
    positional_call = _call_expr("positional")
    keyword_call = _call_expr("keyword")
    or_pattern = OrPattern(
        patterns=[
            ValuePattern(expr=value_call),
            ValuePattern(expr=other_call),
        ],
    )
    mapping_pattern = MappingPattern(
        keys=[mapping_key_call],
        values=[ValuePattern(expr=_call_expr("mapped"))],
        rest=None,
    )
    class_pattern = ClassPattern(
        class_ref=NameExpr("cls"),
        positionals=[
            ValuePattern(expr=positional_call),
            StarredPattern(capture=NameExpr("rest")),
        ],
        keyword_keys=["keyword"],
        keyword_values=[ValuePattern(expr=keyword_call)],
    )
    assert value_call in _iter_call_exprs(or_pattern)
    assert other_call in _iter_call_exprs(or_pattern)
    assert mapping_key_call in _iter_call_exprs(mapping_pattern)
    assert positional_call in _iter_call_exprs(class_pattern)
    assert keyword_call in _iter_call_exprs(class_pattern)


def test_collects_calls_from_as_and_sequence_patterns() -> None:
    """``AsPattern`` and ``SequencePattern`` recurse into nested
    patterns.
    """
    inner_call = _call_expr("inner")
    as_pattern = AsPattern(
        pattern=SequencePattern(
            patterns=[ValuePattern(expr=inner_call)],
        ),
        name=NameExpr("capture"),
    )
    calls = _iter_call_exprs(as_pattern)
    assert inner_call in calls


def test_collects_calls_from_mapping_pattern_with_rest() -> None:
    """``MappingPattern`` collects calls from keys and ``rest``
    bindings.
    """
    key_call = _call_expr("key")
    pattern = MappingPattern(
        keys=[key_call],
        values=[ValuePattern(expr=_call_expr("value"))],
        rest=NameExpr("rest"),
    )
    calls = _iter_call_exprs(pattern)
    assert key_call in calls


def test_collects_calls_from_starred_pattern_without_capture() -> None:
    """``StarredPattern`` without a capture name collects no calls."""
    calls = _iter_call_exprs(StarredPattern(capture=None))
    assert calls == []


def test_collects_calls_from_singleton_pattern() -> None:
    """Singleton patterns contain no nested call expressions."""
    calls = _iter_call_exprs(SingletonPattern(value=True))
    assert calls == []
