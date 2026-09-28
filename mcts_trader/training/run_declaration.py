"""TASK_922 creation record for train_jax_fast's actual checkpoint directory path.

This hook does not cover every ad hoc experiment/run creator. It neither infers tasks
from names nor retrofits existing runs. Comment-only PROTECTED.txt means self-protected
under the bound GC policy, not consent to deletion; directive-bearing legacy manifests
keep their 881E meaning.
"""

from datetime import datetime, timezone
from pathlib import Path


def create_protected_run_dir(path, *, name: str, task: str, reason: str) -> bool:
    """Create a new run and its UTF-8 declaration; return False for an existing directory.

    Validate before any mkdir. Only the caller winning the atomic leaf mkdir owns
    declaration creation; a concurrent/existing run and all its manifests stay untouched.
    Exclusive file creation and write/close errors propagate: never report success for a
    missing declaration. A failed new directory is preserved for operator inspection.
    """
    metadata = {"name": name, "task": task, "reason": reason}
    for key, value in metadata.items():
        if not isinstance(value, str) or not value.strip() or not value.isprintable():
            raise ValueError(f"{key} must be nonempty, printable single-line text")
    date = datetime.now(timezone.utc).date().isoformat()
    text = (f"# name: {name}\n# task: {task}\n# date: {date}\n"
            f"# reason: {reason}\n")
    directory = Path(path)
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()  # No exist_ok: atomic ownership, not an exists()/mkdir race.
    except FileExistsError:
        if directory.is_dir():
            return False
        raise
    with (directory / "PROTECTED.txt").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return True
