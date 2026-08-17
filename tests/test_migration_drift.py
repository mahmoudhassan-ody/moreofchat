"""The guard from Task 6's finding.

`conversations` and `usage_ledger` are created in raw SQL. If either ever falls
out of Base.metadata again, autogenerate reads it as dropped and emits
`drop_table('usage_ledger')` — a generated migration that deletes the table a
tenant is invoiced from. The CI job runs autogenerate and asserts the result is
empty; this is the part of it that has logic worth testing.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_migration_drift.py"

_spec = importlib.util.spec_from_file_location("check_migration_drift", SCRIPT)
drift = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drift)


EMPTY = '''
"""ci drift check"""
from alembic import op

revision = "abcd"
down_revision = "0004"


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
'''

DROPS_THE_LEDGER = '''
from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    op.drop_index("ix_usage_ledger_tenant_created", table_name="usage_ledger")
    op.drop_table("usage_ledger")


def downgrade() -> None:
    pass
'''


def test_empty_migration_has_no_findings():
    assert drift.find_operations(EMPTY) == []


def test_docstring_only_body_counts_as_empty():
    source = EMPTY.replace("    pass\n", "")
    assert drift.find_operations(source) == []


def test_drop_table_is_reported():
    findings = drift.find_operations(DROPS_THE_LEDGER)
    assert any("drop_table" in f for f in findings)
    assert any("usage_ledger" in f for f in findings)


def test_every_operation_is_reported_not_just_the_first():
    findings = drift.find_operations(DROPS_THE_LEDGER)
    assert len(findings) == 2


def test_create_table_is_reported_too():
    """Drift in either direction means the models and the database disagree."""
    source = EMPTY.replace("    pass\n", '    op.create_table("whatever")\n', 1)
    assert any("create_table" in f for f in findings) if (
        findings := drift.find_operations(source)
    ) else pytest.fail("create_table not reported")


def test_missing_upgrade_function_is_an_error():
    with pytest.raises(ValueError, match="upgrade"):
        drift.find_operations("x = 1\n")


def test_findings_name_the_line_number():
    findings = drift.find_operations(DROPS_THE_LEDGER)
    assert any(":" in f for f in findings)
