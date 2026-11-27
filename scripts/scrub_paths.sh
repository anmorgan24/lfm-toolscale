#!/usr/bin/env bash
# Replace home-directory paths (a macOS or Linux home folder) with ~ in results, data, and logs,
# so local usernames never reach a commit. Run before `git add` after any new run.
set -euo pipefail
cd "$(dirname "$0")/.."
files=$(grep -rlE "/(Users|home)/[^/~ ]+" results data 2>/dev/null || true)
if [ -z "$files" ]; then echo "No home-directory paths found."; exit 0; fi
for f in $files; do
  perl -pi -e 's#/(Users|home)/[^/\s"]+#~#g' "$f"
  echo "scrubbed: $f"
done
