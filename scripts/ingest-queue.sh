#!/usr/bin/env bash
# Batch-ingest a queue of YouTube URLs into notetaker.
#
# Usage:
#   ./ingest-queue.sh path/to/queue.txt
#
# Queue format — one URL per line, optional title after a tab:
#   https://youtube.com/watch?v=AAA	Custom Title For This Lecture
#   https://youtube.com/watch?v=BBB
#   # lines starting with # are ignored
#
# Behavior: submits URLs serially to notetaker with a short stagger so the
# server's MAX_CONCURRENT_JOBS=2 semaphore isn't slammed. Failures are logged
# and skipped — queue continues. Appends job_ids to $OUT_DIR/job-ids.txt.

set -uo pipefail  # note: no -e, we want to continue past failures

QUEUE_FILE="${1:?usage: ingest-queue.sh <queue.txt>}"
OUT_DIR="${OUT_DIR:-/tmp/notetaker-ingest}"
mkdir -p "$OUT_DIR"

STAGGER_SECONDS="${STAGGER_SECONDS:-30}"  # gap between submissions

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_IDS_FILE="$OUT_DIR/job-ids.txt"
FAILURES_FILE="$OUT_DIR/failures.txt"

echo "[queue] Reading $QUEUE_FILE"
echo "[queue] Job IDs → $JOB_IDS_FILE"
echo "[queue] Failures → $FAILURES_FILE"

# Pre-flight: confirm notetaker server is reachable before wasting queue time.
NOTETAKER_URL="${NOTETAKER_URL:-http://localhost:8000}"
if [[ -z "${NOTETAKER_PASSWORD:-}" ]]; then
  ENV_FILE="$SCRIPT_DIR/../.env"
  if [[ -f "$ENV_FILE" ]]; then
    NOTETAKER_PASSWORD="$(grep -E '^NOTETAKER_PASSWORD=' "$ENV_FILE" | head -1 | cut -d'=' -f2- | tr -d '"')"
  fi
fi
AUTH_ARG=()
[[ -n "${NOTETAKER_PASSWORD:-}" ]] && AUTH_ARG=(-u "notetaker:${NOTETAKER_PASSWORD}")
echo "[queue] Pre-flight: checking $NOTETAKER_URL"
HTTP_CODE=$(curl -sS -o /dev/null -w "%{http_code}" -m 5 "${AUTH_ARG[@]}" "$NOTETAKER_URL/api/jobs" 2>/dev/null)
if [[ "$HTTP_CODE" != "200" ]]; then
  echo "ERROR: notetaker server at $NOTETAKER_URL returned HTTP $HTTP_CODE (expected 200)" >&2
  echo "       Start the server with:" >&2
  echo "         PIPELINE_TIMEOUT_SECONDS=7200 UPLOAD_RATE_LIMIT=50 uv run notetaker" >&2
  exit 2
fi
echo "[queue] Pre-flight OK."

SUCCESS=0
FAIL=0

while IFS=$'\t' read -r URL TITLE_OVERRIDE; do
  # skip blank lines and comments
  [[ -z "${URL// }" ]] && continue
  [[ "$URL" =~ ^[[:space:]]*# ]] && continue

  echo ""
  echo "=============================================================="
  echo "[queue] Processing: $URL"
  echo "=============================================================="

  if JOB_ID="$("$SCRIPT_DIR/ingest-youtube.sh" "$URL" "${TITLE_OVERRIDE:-}" 2>&1 | tee /dev/stderr | tail -1)"; then
    if [[ "$JOB_ID" =~ ^[a-f0-9]+$ ]]; then
      # Note: ingest-youtube.sh writes to $JOB_IDS_FILE itself, so no duplicate
      # append here.
      SUCCESS=$((SUCCESS + 1))
      echo "[queue] OK: $JOB_ID"
    else
      echo "$URL	bad-job-id: $JOB_ID" >> "$FAILURES_FILE"
      FAIL=$((FAIL + 1))
      echo "[queue] FAIL: could not parse job id"
    fi
  else
    echo "$URL	ingest-script-failed" >> "$FAILURES_FILE"
    FAIL=$((FAIL + 1))
    echo "[queue] FAIL: ingest script returned non-zero"
  fi

  echo "[queue] Sleeping ${STAGGER_SECONDS}s before next submission..."
  sleep "$STAGGER_SECONDS"
done < "$QUEUE_FILE"

echo ""
echo "[queue] Done. $SUCCESS ok, $FAIL failed."
echo "[queue] Tail job status with:"
echo "    for id in \$(cut -f1 $JOB_IDS_FILE); do curl -sS -u notetaker:\$NOTETAKER_PASSWORD http://localhost:8000/api/status/\$id | jq '{id:\"'\$id'\", status, stage, progress}'; done"
