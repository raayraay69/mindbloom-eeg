# MindBloom Validation Ralph Loop

Multi-agent parallel execution system for validating the MindBloom EEG schizophrenia screening project.

## Overview

This Ralph loop implementation orchestrates multiple Claude agents in parallel to complete the four validation phases defined in the research paper (Section 6.3).

## Structure

```
ralph/
├── RALPH_PROMPT.md           # Main orchestration prompt
├── PROGRESS.md               # Phase status tracker
├── config.json               # Orchestration configuration
├── ralph-loop.sh             # Main loop runner
├── run-parallel-agents.sh    # Multi-agent spawner
├── agents/                   # Phase-specific agent prompts
│   ├── phase1-internal-validation.md
│   ├── phase2-hardware-validation.md
│   ├── phase3-external-validation.md
│   ├── phase4-prospective-pilot.md
│   └── webapp-enhancements.md
├── hooks/
│   └── stop-on-complete.sh   # Completion detection hook
└── logs/                     # Agent execution logs
```

## Validation Phases

| Phase | Name | Dependencies | Parallel Safe |
|-------|------|--------------|---------------|
| 1 | Internal Validation | None | Yes |
| 2 | Hardware Validation | None | Yes |
| 3 | External Validation | Phase 1 | No |
| 4 | Prospective Pilot | Phase 2, 3 | No |
| W | WebApp Enhancements | None | Yes |

## Execution Matrix

**Stage 1** (parallel): Phase 1, Phase 2, WebApp
**Stage 2** (after Phase 1): Phase 3
**Stage 3** (after Phase 2+3): Phase 4

## Usage

### Option 1: Full Ralph Loop

Runs continuously until all phases complete or max iterations reached:

```bash
cd /Users/cash/Desktop/fifi/ralph
./ralph-loop.sh
```

Configuration via environment variables:
```bash
MAX_ITERATIONS=50 PARALLEL_AGENTS=3 ./ralph-loop.sh
```

### Option 2: Single Parallel Execution

Spawns all eligible agents once based on current progress:

```bash
./run-parallel-agents.sh
```

### Option 3: Manual Agent Execution

Run a specific phase agent directly:

```bash
cat agents/phase1-internal-validation.md | claude --print
```

## Completion Criteria

The loop terminates when PROGRESS.md contains:
```
RALPH_LOOP_COMPLETE: All validation phases finished successfully.
```

Individual phase completion signals:
- `PHASE1_COMPLETE`: Internal validation passed (>=80% accuracy)
- `PHASE2_COMPLETE`: Hardware validated (SNR acceptable, latency <50ms)
- `PHASE3_COMPLETE`: External validation done (>=70% or domain shift documented)
- `PHASE4_COMPLETE`: IRB materials ready, clinical site identified
- `WEBAPP_COMPLETE`: High-priority enhancements deployed

## Progress Tracking

Check current status:
```bash
cat PROGRESS.md
```

View agent logs:
```bash
ls -la logs/
cat logs/phase1-internal-validation.log
```

## Safety Features

1. **Max iterations**: Default 50; prevents infinite loops
2. **Dependency enforcement**: Phase 3 cannot start until Phase 1 completes
3. **Graceful interrupts**: Ctrl+C cleanly stops all agents
4. **Logging**: All agent output captured to `logs/`

## Editorial Standards

All agents follow the project's editorial guidelines:
- Graduate-level academic tone
- No em dashes (use commas, parentheses, semicolons)
- No cliches or hedge words
- Active voice preferred

## Troubleshooting

**Agents not spawning**: Check dependencies in `config.json`
**Loop not terminating**: Verify completion phrase matches exactly
**Phase stuck**: Check agent log for errors; review PROGRESS.md

## Integration with Claude Code

Run from within Claude Code session:
```
/cd ralph
/run ./ralph-loop.sh
```

Or spawn agents manually using Task tool with agent prompts.
