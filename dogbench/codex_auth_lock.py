"""Cross-process serialization for Codex CLI calls sharing ChatGPT auth.

Codex rotates refresh tokens.  Two isolated agent processes that start from the
same auth.json can otherwise both try to persist different successors, leaving
one process with a token the server has already invalidated.
"""
from __future__ import annotations

import fcntl
import functools
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


def lock_path() -> Path:
    override = os.environ.get("DOCBENCH_CODEX_AUTH_LOCK")
    if override:
        return Path(override).expanduser()
    configured_auth = os.environ.get("DOCBENCH_CODEX_AUTH_FILE", "").strip()
    if configured_auth:
        return Path(configured_auth).expanduser().parent / "docbench-auth.lock"
    return Path(tempfile.gettempdir()) / f"dogbench-{os.getuid()}-codex-auth.lock"


@contextmanager
def codex_auth_lock() -> Iterator[None]:
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def serialize_codex_auth(func: Callable[P, R]) -> Callable[P, R]:
    """Decorate one complete Codex invocation, including auth copy/persist."""

    @functools.wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with codex_auth_lock():
            return func(*args, **kwargs)

    return wrapped
