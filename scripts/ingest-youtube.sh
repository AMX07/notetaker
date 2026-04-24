#!/usr/bin/env bash
# Download a YouTube video with yt-dlp and submit it to notetaker's /api/convert.
#
# Usage:
#   ./ingest-youtube.sh "https://youtube.com/watch?v=XXXX" [optional raw title override]
#
# Prereqs (once):
#   brew install yt-dlp
#   notetaker server running at $NOTETAKER_URL (default http://localhost:8000)
#   notetaker started with bulk-run env:
#     PIPELINE_TIMEOUT_SECONDS=7200 UPLOAD_RATE_LIMIT=50 uv run notetaker
#
# Env (optional):
#   NOTETAKER_URL       — base URL, default http://localhost:8000
#   NOTETAKER_PASSWORD  — HTTP Basic password, read from notetaker/.env if unset
#   DOWNLOAD_DIR        — where yt-dlp writes mp4, default /tmp/notetaker-ingest

set -euo pipefail

URL="${1:?usage: ingest-youtube.sh <youtube-url> [raw-title]}"
RAW_TITLE_OVERRIDE="${2:-}"

NOTETAKER_URL="${NOTETAKER_URL:-http://localhost:8000}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tmp/notetaker-ingest}"
mkdir -p "$DOWNLOAD_DIR"

# Load NOTETAKER_PASSWORD from .env if not already set
if [[ -z "${NOTETAKER_PASSWORD:-}" ]]; then
  ENV_FILE="$(dirname "$0")/../.env"
  if [[ -f "$ENV_FILE" ]]; then
    NOTETAKER_PASSWORD="$(grep -E '^NOTETAKER_PASSWORD=' "$ENV_FILE" | head -1 | cut -d'=' -f2- | tr -d '"')"
  fi
fi

if ! command -v yt-dlp >/dev/null 2>&1; then
  echo "ERROR: yt-dlp not installed. Run: brew install yt-dlp" >&2
  exit 127
fi

echo "[ingest] Fetching metadata for $URL"
RAW_TITLE="${RAW_TITLE_OVERRIDE:-$(yt-dlp --print "%(title)s" --no-warnings "$URL")}"
echo "[ingest] Raw title: $RAW_TITLE"

# clean_title normalizes YouTube's noisy raw title into "<Topic> - <Speaker>".
# Strips YC/Startup School marks (trademark hygiene per the plan's legal notes),
# reorders leading-speaker patterns to topic-first, drops trailing year-parens,
# normalizes "by <Name>" → "- <Name>", and caps length at 70 chars.
#
# Verified against:
#   "How to Get and Evaluate Startup Ideas - Jared Friedman | YC Startup School 2023"
#     → "How to Get and Evaluate Startup Ideas - Jared Friedman"
#   "Garry Tan - Design For Startups Part 1 (2023)"
#     → "Design For Startups Part 1 - Garry Tan"
#   "Y Combinator Startup School: How to Build Product by Michael Seibel"
#     → "How to Build Product - Michael Seibel"
#
# Override from the CLI when auto-clean produces something wrong:
#   ./ingest-youtube.sh "$URL" "Better Title - Speaker Name"
clean_title() {
  local raw="$1"
  local t="$raw"

  # 1. Strip trailing "| YC Startup School 2023" / ": Y Combinator" / etc.
  t=$(printf '%s' "$t" | sed -E 's/[[:space:]]*[|:-][[:space:]]*(YC[[:space:]]+Startup[[:space:]]+School|Y[[:space:]]+Combinator[[:space:]]+Startup[[:space:]]+School|Y[[:space:]]+Combinator|Startup[[:space:]]+School|YC)([[:space:]]+[0-9]{4})?[[:space:]]*$//')

  # 2. Strip leading "Y Combinator Startup School: " / "YC | " / etc.
  t=$(printf '%s' "$t" | sed -E 's/^(YC[[:space:]]+Startup[[:space:]]+School|Y[[:space:]]+Combinator[[:space:]]+Startup[[:space:]]+School|Y[[:space:]]+Combinator|Startup[[:space:]]+School|YC)[[:space:]]*[|:-][[:space:]]*//')

  # 3. Normalize trailing " by FirstName LastName" → " - FirstName LastName".
  t=$(printf '%s' "$t" | sed -E 's/[[:space:]]+by[[:space:]]+([A-Z][a-zA-Z]+([[:space:]]+[A-Z][a-zA-Z]+){1,2})[[:space:]]*$/ - \1/')

  # 4. Strip trailing "(2023)" or any paren group containing a year.
  t=$(printf '%s' "$t" | sed -E 's/[[:space:]]*\([^)]*[0-9]{4}[^)]*\)[[:space:]]*$//')

  # 5. If starts with "FirstName LastName - " and rest looks like a topic (>20
  #    chars), move speaker to the end. Guarded to avoid false-positives on
  #    3-word topic starts like "Build Something Great - Paul Graham".
  if [[ "$t" =~ ^([A-Z][a-z]+[[:space:]]+[A-Z][a-z]+)[[:space:]]+-[[:space:]]+(.+)$ ]]; then
    local maybe_speaker="${BASH_REMATCH[1]}"
    local rest="${BASH_REMATCH[2]}"
    if [[ ${#rest} -gt 20 && ${#maybe_speaker} -le 25 ]]; then
      t="$rest - $maybe_speaker"
    fi
  fi

  # 6. Collapse internal whitespace + trim edges.
  t=$(printf '%s' "$t" | tr -s ' ' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')

  # 7. Cap at 70 chars with ellipsis.
  if [[ ${#t} -gt 70 ]]; then
    t="${t:0:67}..."
  fi

  printf '%s\n' "$t"
}

TITLE="$(clean_title "$RAW_TITLE")"
echo "[ingest] Cleaned title: $TITLE"

# Download — 720p MP4 max, single-file output, skip re-downloads.
SAFE_STEM="$(echo "$TITLE" | tr -cd '[:alnum:] _-' | tr ' ' '_' | cut -c1-80)"
OUTPUT_PATH="$DOWNLOAD_DIR/${SAFE_STEM}.mp4"

if [[ -f "$OUTPUT_PATH" ]]; then
  echo "[ingest] Cached: $OUTPUT_PATH"
else
  echo "[ingest] Downloading to $OUTPUT_PATH (timeout 10min)"
  timeout 600 yt-dlp \
    -f "best[height<=720][ext=mp4]/best[ext=mp4]/best" \
    --merge-output-format mp4 \
    --no-overwrites \
    --quiet --progress \
    -o "$OUTPUT_PATH" \
    "$URL" || {
      echo "ERROR: yt-dlp timed out or failed for $URL" >&2
      exit 1
    }
fi

if [[ ! -s "$OUTPUT_PATH" ]]; then
  echo "ERROR: download produced empty file at $OUTPUT_PATH" >&2
  exit 1
fi

SIZE_MB=$(( $(stat -f%z "$OUTPUT_PATH" 2>/dev/null || stat -c%s "$OUTPUT_PATH") / 1024 / 1024 ))
echo "[ingest] Downloaded: ${SIZE_MB}MB"

# Submit to notetaker
AUTH_ARG=()
if [[ -n "${NOTETAKER_PASSWORD:-}" ]]; then
  AUTH_ARG=(-u "notetaker:${NOTETAKER_PASSWORD}")
fi

echo "[ingest] Submitting to $NOTETAKER_URL/api/convert"
RESPONSE="$(curl -sS --fail-with-body \
  "${AUTH_ARG[@]}" \
  -F "video=@${OUTPUT_PATH};type=video/mp4" \
  -F "title=${TITLE}" \
  -F "language=en" \
  "$NOTETAKER_URL/api/convert")"

JOB_ID="$(echo "$RESPONSE" | python3 -c 'import sys, json; print(json.load(sys.stdin)["job_id"])')"

# Record the mapping so build-site.py can wire up "Original on YouTube" links.
# Shares the file with ingest-queue.sh — both write to the same path.
JOB_IDS_FILE="${DOWNLOAD_DIR}/job-ids.txt"
printf '%s\t%s\n' "$JOB_ID" "$URL" >> "$JOB_IDS_FILE"

echo "[ingest] Job: $JOB_ID"
echo "[ingest] Mapping appended to $JOB_IDS_FILE"
# Last line is the raw job_id — ingest-queue.sh parses this via tee + tail -1.
echo "$JOB_ID"
