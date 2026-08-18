"""A frozen fixture must be reproducible, not merely labelled frozen.

Both manifests instruct the reader to regenerate rather than hand-edit. That
instruction was unfollowable: the scripts imported pandas, which is not a
project dependency, and read absolute paths under /mnt/user-data/uploads,
which exist on nobody's machine. A build nobody can run is a fixture nobody
can regenerate, and "frozen" quietly degrades into "whatever is committed".

This runs each build script into a temporary directory and compares the output
byte for byte against the committed artifact. Both builds are seeded and
deterministic by design, so it holds — and when it stops holding, the fixture
has drifted from its sources and every case citing it is measuring something
other than what its author wrote.

**Deliberately not skipped when the sources are absent.** A missing `source/`
directory is the failure this file exists to catch, and a skip would restore
exactly the silent hole it was written to close.
"""

import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parents[2] / "evals" / "fixtures"

CASES = [
    pytest.param(FIXTURES / "sinai_demo", "chunks.jsonl", id="sinai_demo"),
    pytest.param(
        FIXTURES / "broker_demo_2026_08_01", "units.jsonl", id="broker_demo_2026_08_01"
    ),
]


@pytest.mark.eval
@pytest.mark.parametrize(("fixture", "artifact"), CASES)
def test_the_sources_are_committed_beside_the_build_script(fixture, artifact):
    """Provenance has to exist to be provenance.

    A figure asserted by an eval case should trace to a row in a file anyone
    can open. Sources living only on the machine that built the fixture make
    every such figure unverifiable the moment that machine is unavailable.
    """
    source = fixture / "source"
    assert source.is_dir(), f"{source} is missing — the fixture has no provenance"
    assert list(source.glob("*.csv")), f"{source} contains no CSVs"


@pytest.mark.eval
@pytest.mark.parametrize(("fixture", "artifact"), CASES)
def test_the_fixture_rebuilds_byte_identically(fixture, artifact, tmp_path):
    """Run the build in a temp directory; the output must match what is committed.

    A subprocess rather than an import: the scripts are standalone programs
    with module-level side effects, and running one the way a person would is
    the only way to prove a person can.
    """
    build = fixture / "build.py"
    result = subprocess.run(  # noqa: S603 - a repo-local script, no user input
        [sys.executable, str(build)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{build.name} failed:\n{result.stdout}\n{result.stderr}"
    )

    rebuilt = (tmp_path / artifact).read_bytes()
    committed = (fixture / artifact).read_bytes()
    assert rebuilt == committed, _explain(rebuilt, committed, fixture, artifact)


def _explain(rebuilt: bytes, committed: bytes, fixture: Path, artifact: str) -> str:
    """Name the first differing line rather than dumping two files at someone."""
    new = rebuilt.decode("utf-8").splitlines()
    old = committed.decode("utf-8").splitlines()
    if len(new) != len(old):
        return (
            f"{fixture.name}/{artifact} rebuilt with {len(new)} lines, "
            f"committed has {len(old)} — the fixture has drifted from its sources"
        )
    for number, (a, b) in enumerate(zip(new, old, strict=True), start=1):
        if a != b:
            return (
                f"{fixture.name}/{artifact} differs at line {number}.\n"
                f"  rebuilt:   {a[:200]}\n"
                f"  committed: {b[:200]}\n"
                f"Either the sources changed or the build did. Both mean every "
                f"case citing this fixture is now asserting against something else."
            )
    return "byte lengths differ with identical lines — check trailing newlines"


@pytest.mark.eval
@pytest.mark.parametrize(("fixture", "artifact"), CASES)
def test_the_build_script_needs_no_dependency_the_project_lacks(fixture, artifact):
    """Stdlib only.

    pandas was not in the dependency tree and adding it to run two small CSV
    readers would be a dependency the application never uses, carried for the
    life of the project. The guard is here rather than in a comment because
    the next person to touch these scripts will reach for pandas by reflex.
    """
    source = (fixture / "build.py").read_text(encoding="utf-8")
    body = "\n".join(
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    )
    assert "pandas" not in body, f"{fixture.name}/build.py imports pandas again"
    assert "numpy" not in body
