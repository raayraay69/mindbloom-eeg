#!/bin/bash
# MindBloom Validation Ralph Loop
# Multi-agent parallel execution for validation phases

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROMPT_FILE="$SCRIPT_DIR/RALPH_PROMPT.md"
PROGRESS_FILE="$SCRIPT_DIR/PROGRESS.md"
LOG_FILE="$SCRIPT_DIR/ralph.log"

# Configuration
MAX_ITERATIONS=${MAX_ITERATIONS:-50}
COMPLETION_PHRASE="RALPH_LOOP_COMPLETE"
PARALLEL_AGENTS=${PARALLEL_AGENTS:-3}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[$timestamp]${NC} $1"
    echo "[$timestamp] $1" >> "$LOG_FILE"
}

log_success() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}[$timestamp] ✓${NC} $1"
    echo "[$timestamp] SUCCESS: $1" >> "$LOG_FILE"
}

log_warning() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}[$timestamp] ⚠${NC} $1"
    echo "[$timestamp] WARNING: $1" >> "$LOG_FILE"
}

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${RED}[$timestamp] ✗${NC} $1"
    echo "[$timestamp] ERROR: $1" >> "$LOG_FILE"
}

check_completion() {
    if grep -q "$COMPLETION_PHRASE" "$PROGRESS_FILE" 2>/dev/null; then
        return 0
    fi
    return 1
}

update_iteration_count() {
    local iteration=$1
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Update the iteration count in PROGRESS.md
    sed -i '' "s/\*\*Loop Iteration\*\*: [0-9]*/\*\*Loop Iteration\*\*: $iteration/" "$PROGRESS_FILE"
    sed -i '' "s/\*\*Last Updated\*\*: .*/\*\*Last Updated\*\*: $timestamp/" "$PROGRESS_FILE"
}

run_orchestrator() {
    local iteration=$1
    log "Running orchestrator iteration $iteration..."

    # Run Claude with the main prompt, allowing parallel agent spawning
    cat "$PROMPT_FILE" | claude --print \
        --allowedTools "Task,Read,Write,Edit,Grep,Glob,Bash,TodoWrite" \
        2>&1 | tee -a "$LOG_FILE"

    return $?
}

# Main loop
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║         MindBloom Validation Ralph Loop                      ║"
    echo "║         Multi-Agent Parallel Execution                       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    log "Starting Ralph Loop"
    log "Max iterations: $MAX_ITERATIONS"
    log "Parallel agents: $PARALLEL_AGENTS"
    log "Prompt file: $PROMPT_FILE"
    log "Progress file: $PROGRESS_FILE"
    echo ""

    # Check prerequisites
    if ! command -v claude &> /dev/null; then
        log_error "Claude CLI not found. Please install it first."
        exit 1
    fi

    if [ ! -f "$PROMPT_FILE" ]; then
        log_error "Prompt file not found: $PROMPT_FILE"
        exit 1
    fi

    # Initialize log
    echo "=== Ralph Loop Started $(date) ===" >> "$LOG_FILE"

    local iteration=0

    while [ $iteration -lt $MAX_ITERATIONS ]; do
        iteration=$((iteration + 1))

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log "ITERATION $iteration of $MAX_ITERATIONS"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        update_iteration_count $iteration

        # Run the orchestrator
        if ! run_orchestrator $iteration; then
            log_warning "Orchestrator returned non-zero exit code"
        fi

        # Check for completion
        if check_completion; then
            echo ""
            log_success "COMPLETION DETECTED!"
            log_success "All validation phases finished successfully."
            echo ""
            echo "╔══════════════════════════════════════════════════════════════╗"
            echo "║                    RALPH LOOP COMPLETE                       ║"
            echo "╚══════════════════════════════════════════════════════════════╝"
            echo ""
            echo "=== Ralph Loop Completed $(date) ===" >> "$LOG_FILE"
            exit 0
        fi

        log "Iteration $iteration complete. Continuing..."

        # Small delay between iterations to prevent rate limiting
        sleep 2
    done

    log_warning "Max iterations ($MAX_ITERATIONS) reached without completion"
    echo "=== Ralph Loop Max Iterations Reached $(date) ===" >> "$LOG_FILE"
    exit 1
}

# Handle interrupts gracefully
trap 'log_warning "Interrupted by user"; exit 130' INT TERM

# Run main
main "$@"
