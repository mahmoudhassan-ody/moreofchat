#!/usr/bin/env python3
"""Fail if `alembic revision --autogenerate` produced a non-empty migration.

Run in CI after `alembic upgrade head`. A non-empty result means the ORM models
and the live schema disagree, and the direction that matters is deletion:
`conversations` and `usage_ledger` are created in raw SQL, so if either ever
falls out of `Base.metadata` again, autogenerate reads it as dropped and writes
`op.drop_table("usage_ledger")` into the next revision somebody runs. That is a
generated migration that deletes the table tenants are invoiced from.

Usage:
    check_migration_drift.py <generated-revision.py>

Exits 0 when the migration is empty, 1 with the offending operations otherwise.
"""

import ast
import sys
from pathlib import Path

_BODIES = ("upgrade", "downgrade")


def find_operations(source: str) -> list[str]:
    """Return every Alembic operation in the migration's upgrade/downgrade bodies.

    An empty list means autogenerate found no difference. Docstrings, `pass`,
    and the comment banners Alembic emits are all ignored; anything else is a
    schema difference the models did not declare.
    """
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in _BODIES
    }
    missing = [name for name in _BODIES if name not in functions]
    if missing:
        raise ValueError(f"migration defines no {' or '.join(missing)}() function")

    findings = []
    for name in _BODIES:
        for statement in functions[name].body:
            if _is_noise(statement):
                continue
            findings.append(f"{name}():{statement.lineno}: {_describe(statement)}")
    return findings


def _is_noise(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Pass):
        return True
    # A bare string expression is the docstring Alembic templates in.
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)


def _describe(statement: ast.stmt) -> str:
    """Name the operation the way it reads in the file, e.g. op.drop_table("x")."""
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        call = statement.value
        name = ast.unparse(call.func)
        arguments = ", ".join(ast.unparse(a) for a in call.args)
        return f"{name}({arguments})"
    return ast.unparse(statement)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    path = Path(argv[1])
    findings = find_operations(path.read_text(encoding="utf-8"))
    if not findings:
        print(f"migration drift check: {path.name} is empty, models match the schema")
        return 0

    print(f"MIGRATION DRIFT: autogenerate produced operations in {path.name}", file=sys.stderr)
    for finding in findings:
        print(f"  {finding}", file=sys.stderr)
    print(
        "\nThe ORM models and the database disagree. If this lists drop_table for a "
        "table created in raw SQL, the model is missing from src/moc/tenancy/models.py "
        "— add it before this revision reaches anyone's database.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
