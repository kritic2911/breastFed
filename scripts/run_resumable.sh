#!/usr/bin/env bash
# scripts/run_resumable.sh
# Relaunches training runs that exit with code 75 (thermal timeout / resumable exit).
# Includes retry cap, cool-down sleep, and auto-resume.

set -euo pipefail

MAX_RETRIES=5
RETRY_DELAY=60
RETRY_COUNT=0

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <command...>"
    exit 1
fi

COMMAND=("$@")

while [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; do
    echo "[run_resumable] Executing: ${COMMAND[*]} (Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
    
    set +e
    "${COMMAND[@]}"
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[run_resumable] Command finished successfully."
        exit 0
    elif [ "$EXIT_CODE" -eq 75 ]; then
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "[run_resumable] Process exited with code 75 (Thermal pause timeout). Cooling down for ${RETRY_DELAY}s..."
        sleep "$RETRY_DELAY"
        
        # Ensure resume auto flag is present for subsequent runs
        if [[ ! " ${COMMAND[*]} " =~ " --resume " ]]; then
            COMMAND+=("--resume" "auto")
        fi
    else
        echo "[run_resumable] Process exited with unhandled error code $EXIT_CODE. Aborting."
        exit "$EXIT_CODE"
    fi
done

echo "[run_resumable] Maximum retries ($MAX_RETRIES) reached. Aborting."
exit 1

