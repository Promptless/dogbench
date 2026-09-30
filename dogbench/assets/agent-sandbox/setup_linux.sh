#!/usr/bin/env bash
# Adapted from the original setup_ec2.sh; operator supplies all resource names.
set -euo pipefail
: "${DOCBENCH_AGENT_SANDBOX_IMAGE:?Set a tag for the agent image you will build}"
: "${DOCBENCH_AGENT_SANDBOX_NETWORK:?Set the dedicated internal network name}"
: "${DOCBENCH_EGRESS_NETWORK:?Set the dedicated outbound proxy network name}"
: "${DOCBENCH_PROXY_NAME:?Set a new proxy container name}"
: "${DOCBENCH_PROXY_IMAGE:?Set the Squid image tag/digest}"
RUNTIME="${DOCBENCH_AGENT_SANDBOX_BACKEND:-docker}"
case "$RUNTIME" in docker|podman) ;; *) echo 'Use docker or podman' >&2; exit 1;; esac
ASSETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v "$RUNTIME" >/dev/null
command -v sudo >/dev/null
if sudo -n "$RUNTIME" container inspect "$DOCBENCH_PROXY_NAME" >/dev/null 2>&1; then
    echo 'Configured proxy container already exists; choose a new name or manage it explicitly.' >&2
    exit 1
fi
sudo -n "$RUNTIME" build --pull -t "$DOCBENCH_AGENT_SANDBOX_IMAGE" -f "$ASSETS_DIR/Containerfile" "$ASSETS_DIR"
sudo -n "$RUNTIME" network inspect "$DOCBENCH_EGRESS_NETWORK" >/dev/null 2>&1 \
    || sudo -n "$RUNTIME" network create "$DOCBENCH_EGRESS_NETWORK"
sudo -n "$RUNTIME" network inspect "$DOCBENCH_AGENT_SANDBOX_NETWORK" >/dev/null 2>&1 \
    || sudo -n "$RUNTIME" network create --internal "$DOCBENCH_AGENT_SANDBOX_NETWORK"
IS_INTERNAL="$(sudo -n "$RUNTIME" network inspect --format '{{.Internal}}' "$DOCBENCH_AGENT_SANDBOX_NETWORK")"
if [ "$IS_INTERNAL" != 'true' ]; then
    echo 'Candidate network must be internal; refusing an existing outbound network.' >&2
    exit 1
fi
sudo -n "$RUNTIME" run -d --name "$DOCBENCH_PROXY_NAME" --restart=always \
    --network "$DOCBENCH_EGRESS_NETWORK" \
    --volume "$ASSETS_DIR/squid.conf:/etc/squid/squid.conf:ro,Z" \
    "$DOCBENCH_PROXY_IMAGE"
sudo -n "$RUNTIME" network connect "$DOCBENCH_AGENT_SANDBOX_NETWORK" "$DOCBENCH_PROXY_NAME"
printf 'Set DOCBENCH_AGENT_EGRESS_PROXY=http://%s:3128\n' "$DOCBENCH_PROXY_NAME"
