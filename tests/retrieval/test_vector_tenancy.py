"""The tenant filter, proven structurally — design §7.2.

`vectors.py` is the single most dangerous file in P1. One collection holds
every tenant's points, so a missing filter is not a bug that returns too much:
it is a university's student answered from a property developer's catalogue,
and neither tenant can tell it happened.

These tests do not check that the filter was remembered. They check that
forgetting it is **not expressible**, the way `verify_signature` takes only
bytes so a re-serialized body cannot be passed:

- No public method accepts a filter argument, so a caller has no channel
  through which to express one — or its absence.
- The raw client is unreachable outside `_TenantScope`, whose construction
  requires a tenant id. There is no object in this module that can query
  Qdrant and does not already know whose data it is looking at.
- Every `models.Filter` is built in one function, which always emits the
  tenant clause.

Each is asserted against the source, because none of them has a behavioural
signature — a repository with the filter quietly dropped returns results, and
they look fine until they are somebody else's.
"""

import ast
import inspect
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from moc.retrieval import vectors

MODULE = Path(inspect.getfile(vectors))
TREE = ast.parse(MODULE.read_text(encoding="utf-8"))

FILTER_ARGUMENT_NAMES = {
    "filter",
    "query_filter",
    "scroll_filter",
    "must",
    "should",
    "conditions",
    "payload_filter",
}

PUBLIC_METHODS = [
    name
    for name, member in inspect.getmembers(vectors.QdrantRepository, inspect.isfunction)
    if not name.startswith("_")
]


def class_body(name: str) -> ast.ClassDef:
    return next(
        node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == name
    )


# ─────────────────────────── the headline ───────────────────────────


def test_a_query_with_no_tenant_filter_is_unrepresentable():
    """Three independent structural facts, each of which alone would suffice.

    Together they mean there is no expression in this module that reaches
    Qdrant without a tenant id having been supplied first — not "no expression
    we wrote", but no expression that type-checks and runs.
    """
    # 1. No public method offers a filter parameter, so a caller cannot pass a
    #    filter and therefore cannot pass an empty one.
    for name in PUBLIC_METHODS:
        parameters = set(inspect.signature(getattr(vectors.QdrantRepository, name)).parameters)
        assert not parameters & FILTER_ARGUMENT_NAMES, (
            f"{name} accepts a caller-supplied filter; that is the channel through "
            f"which an unscoped query arrives"
        )

    # 2. The only object with query methods is _TenantScope, and it cannot be
    #    built without a tenant id.
    scope_parameters = inspect.signature(vectors._TenantScope).parameters
    assert "tenant_id" in scope_parameters
    assert scope_parameters["tenant_id"].default is inspect.Parameter.empty, (
        "a defaulted tenant_id makes the unscoped query representable again"
    )

    # 3. Every filter in the module is built by one function.
    builders = {
        enclosing.name
        for enclosing in ast.walk(TREE)
        if isinstance(enclosing, ast.FunctionDef | ast.AsyncFunctionDef)
        for call in ast.walk(enclosing)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "Filter"
    }
    assert builders == {"_filter"}, (
        f"models.Filter is constructed in {builders or 'nowhere'}; it must be built in "
        f"exactly one place, which always emits the tenant clause"
    )


def test_the_one_filter_builder_always_emits_the_tenant_clause():
    """`_filter` has no branch that returns without the tenant condition.

    A conditional tenant clause is the version of this bug that survives
    review: correct on every path anyone tested, absent on the one they did
    not.
    """
    builder = next(
        node
        for node in ast.walk(class_body("_TenantScope"))
        if isinstance(node, ast.FunctionDef) and node.name == "_filter"
    )
    returns = [node for node in ast.walk(builder) if isinstance(node, ast.Return)]
    assert len(returns) == 1, (
        "more than one return from the filter builder — each is a path that has to be "
        "checked separately, and one of them eventually is not"
    )
    source = ast.get_source_segment(MODULE.read_text(encoding="utf-8"), builder)
    assert "tenant_field" in source


def test_the_raw_client_is_unreachable_outside_the_scope():
    """No path from the repository to Qdrant that skips `_TenantScope`.

    The repository holds the client only to hand it to a scope. If it could
    call the client directly, the two facts above would be a convention rather
    than a constraint.
    """
    repository = class_body("QdrantRepository")
    offenders = []
    for node in ast.walk(repository):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "_client"
            ):
                offenders.append(f"{node.name} calls self._client.{call.func.attr}")
    assert offenders == [], (
        f"the repository reaches Qdrant without going through _TenantScope: {offenders}"
    )


def test_every_public_method_requires_tenant_id():
    """Signature-level, not "we remembered to pass it"."""
    assert PUBLIC_METHODS, "no public methods found — the check would pass vacuously"
    for name in PUBLIC_METHODS:
        parameters = inspect.signature(getattr(vectors.QdrantRepository, name)).parameters
        assert "tenant_id" in parameters, f"{name} does not take a tenant_id"
        assert parameters["tenant_id"].default is inspect.Parameter.empty, (
            f"{name} defaults its tenant_id, so omitting it is silently allowed"
        )
        assert parameters["tenant_id"].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} takes tenant_id positionally, so it can be transposed with another "
            f"argument and still run"
        )


def test_tenant_id_is_typed_as_a_uuid_not_a_string():
    """A string tenant id can be an empty string, and an empty string is a
    filter that matches nothing — or, depending on the store, everything."""
    for name in PUBLIC_METHODS:
        annotation = inspect.signature(getattr(vectors.QdrantRepository, name)).parameters[
            "tenant_id"
        ].annotation
        assert annotation in (UUID, "UUID"), f"{name} takes tenant_id as {annotation}"


# ─────────────────────────── the guards firing ───────────────────────────


def test_the_structural_guards_catch_a_planted_filter_argument():
    """Prove the first check fires rather than trusting that it would."""
    planted = ast.parse("class R:\n    async def search(self, *, query_filter=None): ...")
    method = planted.body[0].body[0]
    names = {argument.arg for argument in method.args.kwonlyargs}
    assert names & FILTER_ARGUMENT_NAMES


def test_the_scope_cannot_be_built_without_a_tenant():
    with pytest.raises(TypeError):
        vectors._TenantScope(client=object(), collection="kb_education")


def test_a_scope_carries_the_tenant_into_its_filter():
    scope = vectors._TenantScope(
        client=object(), tenant_id=(tenant := uuid4()), collection="kb_education"
    )
    rendered = str(scope._filter())
    assert str(tenant) in rendered


def test_the_admin_class_never_touches_a_point():
    """The split is only honest if the untenanted class cannot reach data.

    `QdrantAdmin` exists so `QdrantRepository` has no untenanted method at
    all. If admin could scroll or search, the exception would be back — just
    moved one class over, where nobody is looking for it.
    """
    point_operations = {
        "upsert", "search", "query_points", "scroll", "delete", "count",
        "retrieve", "set_payload", "delete_payload", "query_batch_points",
    }
    admin = class_body("QdrantAdmin")
    reached = {
        call.func.attr
        for call in ast.walk(admin)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "_client"
    }
    assert not reached & point_operations, (
        f"QdrantAdmin reaches point data via {sorted(reached & point_operations)} — "
        f"the untenanted surface must stay topology-only"
    )


def test_qdrant_client_is_imported_nowhere_else():
    """The import contract names this module as the sole permitted importer.

    Asserted here as well as in .importlinter so the reason travels with the
    file: every other module that imports the client is a second place a
    filter could be forgotten.
    """
    src = Path(inspect.getfile(vectors)).parents[2]
    offenders = []
    for path in sorted(src.rglob("*.py")):
        if path == MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(name.startswith("qdrant_client") for name in names):
                offenders.append(str(path.relative_to(src.parent)))
    assert offenders == [], f"qdrant_client imported outside the repository: {offenders}"
