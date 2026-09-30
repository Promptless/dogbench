# Known issues and reproduction limits

This release extracts the original execution and scoring behavior. It does not
correct benchmark-methodology bugs or regenerate any task context or rubric.

## Contamination checks can flag benign trace content

The public trace scanner can interpret strings inside tool output as network
actions. For example, a local file read that returns a `wget` example, or a
process listing containing `web_search="disabled"`, can trigger a finding.
These false positives were reproduced against the source implementation and
are retained here. A trace finding needs examination of the event that
produced it. Historical trace integrity does not imply a clean scan with every
version of the detector.

## Context checks can reject approved development inputs

The source-identity and solution-prose checks can reject task context containing
wording from a public source title even when that context belongs to the
approved dataset export. A failed preparation is not evidence that the release
file's hash is wrong. The extraction retains the original check; it does not
silently rewrite approved context to get past it.

## Evidence validation does not distinguish added and deleted lines

The original judge validator checks whether quoted evidence occurs in the
patch text. A quote from a deleted line can satisfy that check. The original
Terra prompt, schema, evidence validator, and numeric scorer are preserved,
including this limitation. Numeric parity does not establish that every
criterion judgment is correct.

## Historical artifacts have different coverage and versions

The 735 sanitized trajectories cover 105 public development tasks across seven
historical system configurations. They are not a run of all 175 development
tasks. Some development rubric updates await rescoring. Historical judgments
are not bundled. No new judgments were generated for this extraction, and
missing scores are not reconstructed from nearby versions.

## Runtime validation is bounded

The release tests use synthetic repositories and mocked providers. They do not
run paid evaluations, live cloud agents, or provision external mirror accounts.
Local agent execution requires Linux and the configured container isolation
and model proxy. Provider CLIs, API behavior, authentication, and model
availability can change independently of this snapshot.

Cloud runs require sanitized, attested mirrors and explicit account
configuration. Promptless provisioning, status, and trace retrieval require a
separately supplied private backend. The public adapter fails explicitly when
that backend is unavailable.

The judge's original Codex invocation and retry behavior are retained. Model
selection can be changed explicitly, but that produces a different judge
configuration. The default maximum is 20 attempts per scored patch; live
scoring therefore may involve repeated provider calls.

## Excluded research-suite failures

The reviewed source branch reported four failures in its broader research/site
suite: two Qwen provenance cases, a task-kind normalization expectation, and a
public pairing metadata snapshot. Those research and website workflows are
excluded here. Their failures have not been fixed or represented as passing
release tests. The retained public-package tests are checked independently.
