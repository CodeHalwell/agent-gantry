"""Shared building blocks for Agent-Gantry schema models.

Houses the small pieces that several schema models would otherwise duplicate:
the identifier newline-rejection validator (a security invariant) and the
health-metric fields common to tools and MCP servers.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

__all__ = [
    "HealthMetrics",
    "check_json_constraints",
    "describe_path",
    "json_identity_key",
    "reject_newlines",
    "resolve_numeric_bounds",
    "schema_declares_null",
]


def describe_path(path: str) -> str:
    """Name the thing a validation message is about.

    Every path here is a parameter's, save one: the executor validates the
    argument object *itself* against its root schema's assertions, and that
    path is empty. ``Parameter ''`` names nothing a caller can act on.
    """
    return f"Parameter '{path}'" if path else "The arguments object"


#: The string ``format``s Gantry emits itself, mapped to the Python type each
#: one is reconstructed into. Deliberately not every format the spec names:
#: ``format`` is an *annotation* by default in JSON Schema, and enforcing
#: ``email``/``uri``/``hostname`` on an imported schema that uses them loosely
#: would reject calls that work. These four are different — Gantry emits them
#: precisely because the handler's annotation demands them, so a string that
#: doesn't parse is a call the handler cannot serve.
RECONSTRUCTED_STRING_FORMATS: dict[str, Any] = {
    "date-time": datetime,
    "date": date,
    "time": time,
    "uuid": uuid.UUID,
}

_FORMAT_ADAPTERS: dict[str, Any] = {}


def _format_adapter(fmt: str) -> Any:
    """A memoized ``TypeAdapter`` for one reconstructed string format.

    The *same* parser reconstruction uses, so validation and reconstruction
    agree by construction: previously a malformed ``date-time`` passed
    validation (which read only the JSON type), failed to reconstruct, and the
    fallback handed a `str` to a handler annotated ``datetime`` — reported as
    a success.
    """
    if fmt in _FORMAT_ADAPTERS:
        return _FORMAT_ADAPTERS[fmt]
    python_type = RECONSTRUCTED_STRING_FORMATS.get(fmt)
    if python_type is None:
        return None
    try:
        from pydantic import TypeAdapter

        adapter = TypeAdapter(python_type)
    except Exception:  # noqa: BLE001 - never let this break validation
        adapter = None
    _FORMAT_ADAPTERS[fmt] = adapter
    return adapter


def check_json_constraints(value: Any, schema: dict[str, Any], path: str) -> str | None:
    """Enforce the JSON-Schema constraint keywords, or ``None`` if all hold.

    Covers what Pydantic actually emits for constrained fields and what
    Gantry's own introspection emits: numeric bounds, string length and
    pattern, and array/object length and uniqueness. Keywords whose value is
    the wrong shape are ignored rather than raising — a malformed schema
    should not turn every call into a validation error.

    Each family applies only to values of its own JSON type, as the spec
    requires: ``{"minimum": 5}`` constrains numbers and says nothing about a
    string. That is why this lives here rather than being re-expressed as
    Pydantic metadata in the framework bridge — ``Annotated[Any, Ge(5)]``
    rejects ``"x"``, which the schema permits, so the bridge shares this
    implementation instead of approximating it.
    """
    if isinstance(value, bool):
        return None  # bool is an int subclass; numeric bounds don't apply

    if isinstance(value, (int, float)):
        lower, upper, excl_lower, excl_upper = resolve_numeric_bounds(schema)
        for bound, ok, describe in (
            (lower, lambda v, b: v >= b, "at least"),
            (upper, lambda v, b: v <= b, "at most"),
            (excl_lower, lambda v, b: v > b, "greater than"),
            (excl_upper, lambda v, b: v < b, "less than"),
        ):
            if bound is not None and not ok(value, bound):
                return f"{describe_path(path)} must be {describe} {bound}"
        multiple = schema.get("multipleOf")
        if isinstance(multiple, (int, float)) and not isinstance(multiple, bool) and multiple > 0:
            # Decimal, not ``%``: binary floats make ``0.3 % 0.1`` ≈ 0.1, so a
            # schema-valid JSON number would be rejected. Going through
            # ``str`` gives the decimal the value was written as.
            try:
                divisible = Decimal(str(value)) % Decimal(str(multiple)) == 0
            except (ArithmeticError, ValueError):  # inf/nan and friends
                divisible = True
            if not divisible:
                return f"{describe_path(path)} must be a multiple of {multiple}"

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and not isinstance(min_length, bool):
            if len(value) < min_length:
                return f"{describe_path(path)} must be at least {min_length} characters"
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and not isinstance(max_length, bool):
            if len(value) > max_length:
                return f"{describe_path(path)} must be at most {max_length} characters"
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and pattern:
            try:
                matches = re.search(pattern, value) is not None
            except re.error as exc:
                # Fail open: a pattern Python's ``re`` can't compile is often
                # a valid ECMA-262 one (``\p{L}``), and rejecting every value
                # would break a tool whose schema is fine everywhere else.
                # Logged because the constraint is then silently unenforced —
                # without this the schema's author gets no signal at all.
                logger.warning(
                    "Parameter '%s' declares a pattern %r that Python's re "
                    "cannot compile (%s); the constraint is not enforced.",
                    path,
                    pattern,
                    exc,
                )
                matches = True
            if not matches:
                return f"{describe_path(path)} must match pattern {pattern!r}"
        fmt = schema.get("format")
        if isinstance(fmt, str) and fmt in RECONSTRUCTED_STRING_FORMATS:
            adapter = _format_adapter(fmt)
            if adapter is not None:
                try:
                    adapter.validate_python(value)
                except Exception:  # noqa: BLE001 - any parse failure is a reject
                    return f"{describe_path(path)} must be a valid {fmt} string"

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and not isinstance(min_items, bool):
            if len(value) < min_items:
                return f"{describe_path(path)} must have at least {min_items} items"
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and not isinstance(max_items, bool):
            if len(value) > max_items:
                return f"{describe_path(path)} must have at most {max_items} items"
        if schema.get("uniqueItems") is True:
            # Keyed by JSON identity, not Python equality: ``True == 1`` in
            # Python, but JSON Schema compares types before values, so
            # ``[1, true]`` is two distinct items rather than a duplicate.
            # The keys are hashable, so this is a set rather than a scan.
            seen: set[Any] = set()
            unhashable: list[Any] = []
            for item in value:
                key = json_identity_key(item)
                try:
                    if key in seen:
                        return f"{describe_path(path)} must not contain duplicate items"
                    seen.add(key)
                except TypeError:  # a non-JSON value that isn't hashable
                    if key in unhashable:
                        return f"{describe_path(path)} must not contain duplicate items"
                    unhashable.append(key)

    if isinstance(value, dict):
        # A Pydantic ``dict`` field constrained with ``Field(min_length=1)``
        # emits these, so they arrive inside the inlined mapping schemas —
        # and checking only numbers, strings and arrays let an empty or
        # oversized mapping through to the handler.
        min_properties = schema.get("minProperties")
        if isinstance(min_properties, int) and not isinstance(min_properties, bool):
            if len(value) < min_properties:
                return f"{describe_path(path)} must have at least {min_properties} properties"
        max_properties = schema.get("maxProperties")
        if isinstance(max_properties, int) and not isinstance(max_properties, bool):
            if len(value) > max_properties:
                return f"{describe_path(path)} must have at most {max_properties} properties"

    return None


def null_validates_against(schema: Any) -> bool:
    """Whether ``null`` satisfies ``schema`` — a matching question, not a
    declaring one.

    Distinct from :func:`schema_declares_null`, which asks whether an author
    gave ``null`` a meaning worth preserving. Here a schema that simply never
    forbids null admits it: ``{}`` validates everything, and a constraint-only
    ``{"minimum": 5}`` names no type at all — numeric keywords apply only to
    numbers, so they assert nothing about ``null``.

    Used to count ``oneOf`` branches, where the spec's "exactly one" is
    decided by what matches rather than by what is declared.
    """
    if schema is True:
        return True
    if not isinstance(schema, dict):
        # ``False`` (validates nothing) and any non-schema value.
        return False
    if not schema:
        return True
    # ``const``/``enum`` pin the value outright, whatever the type says.
    if "const" in schema:
        return schema["const"] is None
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return None in enum_values
    declared = schema.get("type")
    if isinstance(declared, str):
        if declared != "null":
            return False
    elif isinstance(declared, list) and "null" not in declared:
        return False
    # Either no ``type`` or one naming null. Every remaining keyword family
    # is typed — numeric bounds, string lengths, array and object shape — so
    # none of them can reject a null. Combinators still can.
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        if not any(null_validates_against(branch) for branch in any_of):
            return False
    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        if sum(1 for branch in one_of if null_validates_against(branch)) != 1:
            return False
    all_of = schema.get("allOf")
    if isinstance(all_of, list) and all_of:
        if not all(null_validates_against(branch) for branch in all_of):
            return False
    return True


def schema_declares_null(prop: Any) -> bool:
    """Whether an explicit ``null`` for this property is a value to keep.

    Callers — the executor's argument normalization and the framework
    ``ToolSpec`` path — are deciding between passing a caller-supplied
    ``None`` through and dropping it as strict mode's "not provided"
    placeholder so the handler's own default applies. The right rule turns
    out to be exactly one question: **keep it iff the executor would accept
    it**, which is :func:`null_validates_against`. Keep a null the schema
    admits and the call is the one the caller asked for; drop one it forbids
    and the default applies instead of a validation error.

    This began as a separate, narrower reading — "did the author *declare*
    null meaningful", where a schema that merely failed to forbid null didn't
    count. That distinction cost more than it bought. It needed a fresh patch
    for each new spelling and still got Gantry's own emission wrong: an
    optional ``Literal["a", None]`` emits ``{"enum": ["a", null]}`` with no
    ``type`` at all, and the declaring reading dropped an explicitly supplied
    ``None``, handing the handler its default instead. It also read a nullable
    ``anyOf`` as nullable while a sibling ``allOf`` forbade null. The matching
    question answers both structurally, because it composes instead of
    laddering. Kept as a distinct name because it is what the *callers* are
    asking (PR #381 review).
    """
    return null_validates_against(prop)


def _numeric(value: Any) -> float | int | None:
    """``value`` when it is a JSON number, else ``None`` (booleans excluded)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def resolve_numeric_bounds(schema: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """A schema's bounds as ``(min, max, exclusiveMin, exclusiveMax)``.

    Two JSON Schema dialects spell exclusivity differently and both reach this
    library. Modern drafts give ``exclusiveMinimum`` a *number*; draft-04 —
    which is what OpenAPI 3.0 emits, and OpenAPI/MCP import is a supported way
    to register a tool — gives it a *boolean* that promotes ``minimum`` to an
    exclusive bound. Reading only the modern form left the boolean ignored and
    the bound applied inclusively, so ``{"minimum": 5, "exclusiveMinimum":
    true}`` accepted ``5``.

    Lives here, beside :func:`json_identity_key`, for the same reason: the
    executor's validator and the framework args-model bridge must read a bound
    identically, or a framework accepts what the engine rejects.
    """
    lower = _numeric(schema.get("minimum"))
    upper = _numeric(schema.get("maximum"))
    excl_lower = schema.get("exclusiveMinimum")
    excl_upper = schema.get("exclusiveMaximum")

    if isinstance(excl_lower, bool):
        excl_lower, lower = (lower, None) if excl_lower else (None, lower)
    else:
        excl_lower = _numeric(excl_lower)
    if isinstance(excl_upper, bool):
        excl_upper, upper = (upper, None) if excl_upper else (None, upper)
    else:
        excl_upper = _numeric(excl_upper)

    return lower, upper, excl_lower, excl_upper


def json_identity_key(value: Any) -> Any:
    """A hashable key equal for exactly the values JSON Schema calls equal.

    Used to enforce ``uniqueItems``. Plain Python equality gets this wrong in
    one specific, reachable way: ``True == 1``, so ``[1, true]`` — two
    distinct instance values in JSON Schema, which compares types before
    values — looked like a duplicate and was rejected. Tagging each value with
    its JSON type separates the two while leaving ``1`` and ``1.0`` equal, as
    JSON Schema requires of numbers.

    Containers are keyed recursively so the same distinction holds inside them
    (``[[1]]`` and ``[[true]]`` are different), and the result is hashable, so
    callers can use a ``set`` rather than a quadratic scan.

    Lives here rather than beside either caller because the executor's
    validator and the framework args-model bridge must agree on it: a
    disagreement means a framework accepts a payload the engine then rejects.
    """
    if isinstance(value, bool):
        # Checked before int — ``bool`` subclasses it.
        return ("boolean", value)
    if isinstance(value, (int, float)):
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    if value is None:
        return ("null", None)
    if isinstance(value, (list, tuple)):
        return ("array", tuple(json_identity_key(item) for item in value))
    if isinstance(value, dict):
        return (
            "object",
            tuple(sorted(((k, json_identity_key(v)) for k, v in value.items()), key=lambda kv: kv[0])),
        )
    # Not a JSON value at all (a handler default, say). Fall back to the value
    # itself; an unhashable one makes the caller drop to an equality scan.
    return ("other", value)


def reject_newlines(value: str | None) -> str | None:
    """Reject newline characters in identifier fields.

    Pydantic v2's Rust regex engine treats ``$`` as end-of-line rather than
    end-of-string, so a ``pattern=r"^...$"`` would accept ``"valid_name\\n"``.
    Attaching this as a reusable ``field_validator`` to every identifier field
    closes that bypass with a single, shared definition.
    """
    if isinstance(value, str) and ("\n" in value or "\r" in value):
        raise ValueError("Value cannot contain newline characters")
    return value


class HealthMetrics(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    """Runtime health fields shared by tools and MCP servers.

    Subclasses add the metrics specific to what they track (per-call latency
    and circuit-breaker state for tools; connection counts and availability for
    MCP servers).
    """

    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    consecutive_failures: int = Field(default=0)
    last_success: datetime | None = None
    last_failure: datetime | None = None
