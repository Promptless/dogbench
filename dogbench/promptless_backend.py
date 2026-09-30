"""Optional boundary for account-specific Promptless operations.

The public adapter submits API triggers and recovers mirror output itself. Account
provisioning, analysis attestations, dispatch status and private trace access are
provided by an operator-owned backend; no private database client is bundled.
Set DOGBENCH_PROMPTLESS_BACKEND to ``package.module:factory`` or pass an instance.
The factory takes no arguments and resolves credentials from its own configuration.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Protocol


class PromptlessBackend(Protocol):
    def ensure_collection(
        self, repo_url: str, *, docs_dir: str | None, analysis_version: str | None,
        timeout_s: int,
    ) -> str:
        """Provision/attest this repository in the explicitly configured account."""
        ...

    def analysis_overlay(
        self, *, repo_path: Path, repo_url: str, collection_id: str,
        current_docs_tree: str,
    ) -> dict[str, Any] | None:
        """Return an attested transient overlay when collection analysis is reused."""
        ...

    def dispatch_status(self, trigger_event_id: str) -> dict[str, Any] | None:
        """Return trigger-keyed status, including suggestion_pr/branch when present.

        The adapter consumes failed, effective_status, trigger_status, detail,
        resolution_reason, suggestion_pr and suggestion_branch. A missing status
        activates the original mirror-polling fallback; it is never completion.
        """
        ...

    def export_trace(self, trigger_event_id: str) -> dict[str, Any] | None:
        """Return this trigger's sanitized trace, or None when unavailable."""
        ...


def load_promptless_backend(backend: PromptlessBackend | None = None) -> PromptlessBackend:
    if backend is None:
        spec = os.environ.get("DOGBENCH_PROMPTLESS_BACKEND", "").strip()
        if not spec:
            raise RuntimeError(
                "Promptless backend is not configured; set DOGBENCH_PROMPTLESS_BACKEND="
                "package.module:factory or pass backend=. Private provisioning, status, "
                "and trace infrastructure are not included in the public package."
            )
        module_name, separator, attribute = spec.partition(":")
        if not separator or not module_name or not attribute:
            raise RuntimeError("DOGBENCH_PROMPTLESS_BACKEND must be package.module:factory")
        try:
            factory = getattr(importlib.import_module(module_name), attribute)
            backend = factory()
        except Exception as exc:
            raise RuntimeError(f"could not initialize configured Promptless backend: {type(exc).__name__}") from exc
    for method in ("ensure_collection", "analysis_overlay", "dispatch_status", "export_trace"):
        if not callable(getattr(backend, method, None)):
            raise RuntimeError(f"configured Promptless backend does not implement {method}()")
    return backend
