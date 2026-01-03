#!/bin/bash
# Multi-Agent Parallel Executor for MindBloom Validation
# Spawns multiple Claude agents in parallel based on dependency graph

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$SCRIPT_DIR/agents"
PROGRESS_FILE="$SCRIPT_DIR/PROGRESS.md"
LOG_DIR="$SCRIPT_DIR/logs"
PIDS_FILE="$SCRIPT_DIR/.running_pids"

mkdir -p "$LOG_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

log_agent() {
    local agent=$1
    local msg=$2
    echo -e "${CYAN}[$(date '+%H:%M:%S')] [$agent]${NC} $msg"
}

# Check if a phase is complete
is_phase_complete() {
    local signal=$1
    grep -q "$signal" "$PROGRESS_FILE" 2>/dev/null
}

# Check if dependencies are met
dependencies_met() {
    local phase=$1
    case $phase in
        "phase1"|"phase2"|"webapp")
            return 0  # No dependencies
            ;;
        "phase3")
            is_phase_complete "PHASE1_COMPLETE"
            return $?
            ;;
        "phase4")
            is_phase_complete "PHASE2_COMPLETE" && is_phase_complete "PHASE3_COMPLETE"
            return $?
            ;;
    esac
    return 1
}

# Run a single agent
run_agent() {
    local agent_name=$1
    local agent_file="$AGENTS_DIR/${agent_name}.md"
    local log_file="$LOG_DIR/${agent_name}.log"

    if [ ! -f "$agent_file" ]; then
        log_agent "$agent_name" "Agent file not found: $agent_file"
        return 1
    fi

    log_agent "$agent_name" "Starting agent..."

    # Run Claude with the agent prompt
    cat "$agent_file" | claude --print \
        --allowedTools "Task,Read,Write,Edit,Grep,Glob,Bash,TodoWrite,WebFetch" \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid:$agent_name" >> "$PIDS_FILE"
    log_agent "$agent_name" "Started with PID $pid"
}

# Wait for all running agents
wait_for_agents() {
    if [ ! -f "$PIDS_FILE" ]; then
        return 0
    fi

    log "Waiting for running agents to complete..."

    while read -r line; do
        local pid=$(echo "$line" | cut -d: -f1)
        local name=$(echo "$line" | cut -d: -f2)

        if ps -p "$pid" > /dev/null 2>&1; then
            log_agent "$name" "Waiting for PID $pid..."
            wait "$pid" 2>/dev/null || true
            log_agent "$name" "Completed"
        fi
    done < "$PIDS_FILE"

    rm -f "$PIDS_FILE"
}

# Determine which agents can run in parallel
get_runnable_agents() {
    local runnable=()

    # Phase 1: Internal Validation
    if ! is_phase_complete "PHASE1_COMPLETE" && dependencies_met "phase1"; then
        runnable+=("phase1-internal-validation")
    fi

    # Phase 2: Hardware Validation
    if ! is_phase_complete "PHASE2_COMPLETE" && dependencies_met "phase2"; then
        runnable+=("phase2-hardware-validation")
    fi

    # Phase 3: External Validation (depends on Phase 1)
    if ! is_phase_complete "PHASE3_COMPLETE" && dependencies_met "phase3"; then
        runnable+=("phase3-external-validation")
    fi

    # Phase 4: Prospective Pilot (depends on Phase 2 and 3)
    if ! is_phase_complete "PHASE4_COMPLETE" && dependencies_met "phase4"; then
        runnable+=("phase4-prospective-pilot")
    fi

    # WebApp Enhancements (no dependencies)
    if ! is_phase_complete "WEBAPP_COMPLETE" && dependencies_met "webapp"; then
        runnable+=("webapp-enhancements")
    fi

    echo "${runnable[@]}"
}

# Main execution
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║       MindBloom Multi-Agent Parallel Executor                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # Clean up any previous PID file
    rm -f "$PIDS_FILE"

    # Get agents that can run
    local agents=($(get_runnable_agents))

    if [ ${#agents[@]} -eq 0 ]; then
        log "No agents available to run. Check dependencies or completion status."
        exit 0
    fi

    log "Runnable agents: ${agents[*]}"
    echo ""

    # Spawn agents in parallel
    for agent in "${agents[@]}"; do
        run_agent "$agent"
    done

    echo ""
    log "All agents spawned. Monitoring progress..."
    echo ""

    # Wait for all to complete
    wait_for_agents

    echo ""
    log "Parallel execution complete. Check logs in $LOG_DIR"

    # Show completion status
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "PHASE STATUS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if is_phase_complete "PHASE1_COMPLETE"; then
        echo -e "  Phase 1 (Internal):   ${GREEN}COMPLETE${NC}"
    else
        echo -e "  Phase 1 (Internal):   ${YELLOW}IN PROGRESS${NC}"
    fi

    if is_phase_complete "PHASE2_COMPLETE"; then
        echo -e "  Phase 2 (Hardware):   ${GREEN}COMPLETE${NC}"
    else
        echo -e "  Phase 2 (Hardware):   ${YELLOW}IN PROGRESS${NC}"
    fi

    if is_phase_complete "PHASE3_COMPLETE"; then
        echo -e "  Phase 3 (External):   ${GREEN}COMPLETE${NC}"
    elif dependencies_met "phase3"; then
        echo -e "  Phase 3 (External):   ${YELLOW}IN PROGRESS${NC}"
    else
        echo -e "  Phase 3 (External):   BLOCKED (waiting for Phase 1)"
    fi

    if is_phase_complete "PHASE4_COMPLETE"; then
        echo -e "  Phase 4 (Pilot):      ${GREEN}COMPLETE${NC}"
    elif dependencies_met "phase4"; then
        echo -e "  Phase 4 (Pilot):      ${YELLOW}IN PROGRESS${NC}"
    else
        echo -e "  Phase 4 (Pilot):      BLOCKED (waiting for Phase 2, 3)"
    fi

    if is_phase_complete "WEBAPP_COMPLETE"; then
        echo -e "  WebApp Enhancements:  ${GREEN}COMPLETE${NC}"
    else
        echo -e "  WebApp Enhancements:  ${YELLOW}IN PROGRESS${NC}"
    fi

    echo ""
}

# Handle Ctrl+C
trap 'log "Interrupted. Cleaning up..."; wait_for_agents; exit 130' INT TERM

main "$@"
