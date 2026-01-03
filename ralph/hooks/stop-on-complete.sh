#!/bin/bash
# Hook to detect completion signals and stop the loop

PROGRESS_FILE="$(dirname "$0")/../PROGRESS.md"

# Check for main completion
if grep -q "RALPH_LOOP_COMPLETE" "$PROGRESS_FILE" 2>/dev/null; then
    echo "COMPLETION_DETECTED: Main loop complete"
    exit 0
fi

# Check individual phase completions and update progress
check_phase() {
    local phase=$1
    local signal=$2

    if grep -q "$signal" "$PROGRESS_FILE" 2>/dev/null; then
        echo "Phase complete: $phase"
    fi
}

check_phase "Phase 1" "PHASE1_COMPLETE"
check_phase "Phase 2" "PHASE2_COMPLETE"
check_phase "Phase 3" "PHASE3_COMPLETE"
check_phase "Phase 4" "PHASE4_COMPLETE"
check_phase "WebApp" "WEBAPP_COMPLETE"

# Continue loop if not all complete
exit 1
