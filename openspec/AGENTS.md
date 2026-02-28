# OpenSpec Agents Workflow

This document describes how AI agents collaborate using OpenSpec in this project.

## Agent Roles

### 1. Codex (Change Designer)
**Location**: Used via CLI (`codex exec`)

**Responsibilities**:
- Create change proposals under `openspec/changes/<change-id>/`
- Generate `tasks.md` with detailed implementation steps
- Create `feature_list.json` for tracking features

**Entry Points**:
```bash
# Interactive mode
codex

# Then use skills:
$openspec-change-interviewer <change-id>
$openspec-feature-list <change-id>
```

### 2. Claude Code (Supervisor)
**Location**: This agent (Claude Code CLI)

**Responsibilities**:
- Supervise implementation via `/monitor-openspec-codex <change-id>`
- Validate worker output and execute tests
- Record evidence and update progress
- Create git checkpoint commits

**Entry Points**:
```bash
# Start monitoring a change
/monitor-openspec-codex <change-id>
```

### 3. Worker (Codex via CLI)
**Location**: Spawned by Supervisor as subagent

**Responsibilities**:
- Implement ONE task at a time
- Create validation bundles under `auto_test_openspec/<change-id>/`
- Write tests (CLI only for this project)
- MUST NOT: toggle checkboxes, declare PASS/FAIL, create git commits

## Workflow Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    User     │────▶│    Codex    │────▶│  Proposals  │
│  (Request)  │     │ (Designer)  │     │ (tasks.md)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                       ┌───────────────────────┘
                       ▼
              ┌─────────────┐     ┌─────────────┐
              │   Claude    │◀───▶│   Worker    │
              │ (Supervisor)│     │  (Codex CLI)│
              └──────┬──────┘     └─────────────┘
                     │
                     ▼
              ┌─────────────┐
              │ Validation  │
              │   Bundle    │
              └─────────────┘
```

## Detailed Steps

### Step 1: Create a Change Proposal (Codex)

1. Open Codex interactive mode:
   ```bash
   codex
   ```

2. Create a change proposal using natural language:
   ```
   I want to add a feature for automatic data poisoning evaluation
   ```

3. Use skill to refine requirements:
   ```
   $openspec-change-interviewer <change-id>
   ```

4. Generate feature list:
   ```
   $openspec-feature-list <change-id>
   ```

### Step 2: Implement Tasks (Claude Code)

1. Open Claude Code in project root

2. Start monitoring:
   ```
   /monitor-openspec-codex <change-id>
   ```

3. Claude Code will:
   - Read `openspec/changes/<change-id>/tasks.md`
   - Pick first eligible unchecked task
   - Spawn Worker (Codex CLI) to implement
   - Validate output and record evidence
   - Toggle checkbox on PASS
   - Continue to next task

## Project-Specific Conventions

### For ML/Data Processing Tasks

- **SCOPE**: Always CLI (no GUI components)
- **Validation**: Python scripts with pytest or direct execution
- **Test Data**: Use small sample datasets in `inputs/`
- **Expected Outputs**: JSON/JSONL with schema validation

### Directory Structure

```
openspec/
├── changes/
│   └── <change-id>/
│       ├── tasks.md           # Implementation tasks
│       ├── feature_list.json  # Feature tracking
│       └── progress.txt       # Handoff log
└── project.md                 # This project spec

auto_test_openspec/
└── <change-id>/
    └── run-<RUN4>__task-<id>__ref-<ref>__<timestamp>/
        ├── task.md            # Run README
        ├── run.sh             # Linux/macOS runner
        ├── run.bat            # Windows runner
        ├── logs/
        │   └── worker_startup.txt
        ├── tests/             # Test scripts
        ├── inputs/            # Test inputs
        ├── outputs/           # Generated outputs
        └── expected/          # Expected outputs
```

### Task Format (tasks.md)

```markdown
- [ ] 1.1 Implement data preprocessing pipeline [#R1]
  - ACCEPT: Pipeline reads raw data, applies IST triggers, outputs JSONL with correct schema
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/<change-id>/<run-folder>/
    - Inputs: inputs/sample_code.jsonl
    - Outputs: outputs/poisoned_code.jsonl
    - Verify: Schema matches expected/poisoned_schema.json
```

### Evidence Format

Worker writes (BUNDLE):
```
BUNDLE (RUN #1): CODEX_CMD=codex exec ... | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/<id>/run-.../ | HOW_TO_RUN: run.sh
```

Supervisor writes (EVIDENCE):
```
EVIDENCE (RUN #1): CODEX_CMD=... | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/<id>/run-.../ | WORKER_STARTUP_LOG: .../logs/worker_startup.txt | VALIDATED_CLI: ./run.sh | EXIT_CODE: 0 | RESULT: PASS | GIT_COMMIT: <sha> | COMMIT_MSG: "..." | FILES: src/..., tests/...
```

## Error Handling

### When a Task is BLOCKED

1. Worker writes under task:
   ```
   BLOCKED: <error excerpt>
   NEEDS: <next concrete step>
   ```

2. Supervisor calls skill:
   ```
   $openspec-unblock-research
   ```

3. Skill produces:
   - Portable unblock report (JSON)
   - Unblock guidance (markdown)

4. Supervisor writes under task:
   ```
   UNBLOCK GUIDANCE (RUN #n): <key conclusions + executable steps>
   ```

### Retry Policy

- MAX_ATTEMPTS = 2 per task
- If blocked: research → retry
- If maxed: mark MAXED, stop if blocking progress

## Governance Rules

### Worker (Codex CLI) MUST NOT:
- Toggle checkboxes in tasks.md
- Write EVIDENCE lines
- Declare PASS/FAIL/RESULT
- Edit feature_list.json
- Create git commits
- Edit runs.log

### Supervisor (Claude Code) MUST:
- Be the ONLY role to toggle checkboxes
- Be the ONLY role to write EVIDENCE lines
- Be the ONLY role to update feature_list.json pass-state
- Be the ONLY role to create git commits
- Validate BEFORE marking PASS

## Commands Reference

### OpenSpec
```bash
openspec init              # Initialize project (already done)
openspec --version         # Check version (0.21.0)
```

### Codex CLI
```bash
codex                      # Interactive mode
codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium "<prompt>"
```

### Claude Code
```
/monitor-openspec-codex <change-id>
```

## Tips for This Project (CausalCode-Defender)

1. **Data Tasks**: When working with data preprocessing, use small sample files (~10-50 records) for validation bundles to keep tests fast.

2. **Model Tasks**: For model training tasks, focus on:
   - Config file correctness
   - Data loader functionality
   - One training step execution (not full training)

3. **IST (Style Transfer)**: Test style application on individual code snippets before full dataset processing.

4. **Defense Evaluation**: Use pre-trained victim models from checkpoints/ directory rather than training from scratch in tests.

5. **Metrics**: Always validate metric computation against known values (e.g., ACC=0.5 for random binary classifier).
