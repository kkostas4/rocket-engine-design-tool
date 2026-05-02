# Noyron-Clone

Reverse-engineered Noyron-style (LEAP 71) computational engine model for a
small methane/LOX rocket engine.

**Status**: scaffold only. Subsystem implementations land via parallel Claude
Code subagent sessions.

## Quick Start

```bash
# Install dependencies
pip install rocketcea numpy scipy matplotlib pytest ruff mypy

# Verify the baseline reference engine reproduces from first principles
python calcs/verify_baseline.py

# Run all subsystem tests (will be empty until subagents land code)
pytest verification/ -v
```

## Reference Engine (Frozen Baseline)

| Parameter | Value |
|---|---|
| Thrust | 1,000 lbf (4,448 N) |
| Pc | 1,000 psia (68.95 bar) |
| Propellants | LCH4 / LOX |
| O/F | 3.60 |
| Expansion ratio | 10.605 |
| Isp delivered | 318.44 s |

Full numbers, source citations, and project rules are in `CLAUDE.md`.

## Working with Claude Code

This project is built around the subagent + parallel-task pattern. See
`.claude/agents/` for specialist definitions and `.claude/commands/` for
slash commands.

### Common workflow
```
/spawn-spine        # build the foundational subsystems in parallel
/verify-design      # re-run all baseline checks + audit the diff
```

### Calling specialists directly
```
"Use the regen-cooling subagent to size the channels at the throat."
"Use the verifier to audit the latest changes to physics/nozzle/."
```

## Layout

```
noyron-clone/
├── CLAUDE.md                 # rules + reference engine numbers
├── physics/                  # analysis (no CAD)
│   ├── thermochem/           # CEA, propellant properties
│   ├── nozzle/               # MOC, Rao
│   ├── regen/                # Bartz, channel sizing
│   ├── injector/             # element sizing, stability
│   ├── structural/           # stress, life
│   └── util/                 # shared helpers (units only, for now)
├── geometry/                 # SDF + CAD generation
│   ├── kernel/
│   ├── chamber/
│   ├── nozzle/
│   └── injector/
├── verification/             # pytest tests + audit reports
├── calcs/                    # one-off Python sanity scripts
└── .claude/
    ├── agents/               # subagent definitions
    ├── commands/             # slash commands
    └── settings.json
```
