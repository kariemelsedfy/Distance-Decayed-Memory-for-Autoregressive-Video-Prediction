# Decision log

This file is append-only. Each decision records the constraint it establishes so
future work does not silently change experimental meaning.

## D-001 — Track A is the only active track

- **Status:** accepted (2026-09-22)
- **Decision:** Track A only; Track B is deferred until the owner reopens it after Gate 2.
- **Consequence:** Do not download Wan checkpoints, prepare Track B data, or use GPU time for Track B.

## D-002 — Start with Memory Maze pixels

- **Status:** accepted (2026-09-22)
- **Decision:** Use Memory Maze 9×9 first, with 64×64 pixels, 4×4 patches, and no VAE.
- **Consequence:** The controlled test remains small and avoids representation-learning confounds from a VAE.

## D-003 — Separate base learning from memory learning

- **Status:** accepted (2026-09-22)
- **Decision:** Train one shared A1 base model, then start every policy-specific A2 memory fine-tune from that checkpoint.
- **Consequence:** Policy comparisons share the same visual and dynamics model and use less compute.

## D-004 — Use streaming A2 training

- **Status:** accepted (2026-09-22)
- **Decision:** A2 uses live caches and a two-chunk gradient window instead of single-pass teacher forcing.
- **Consequence:** Training matches inference cache behavior with linear cost, while older K/V entries are detached and may become stale.

## D-005 — Match budgets as fractions of the horizon

- **Status:** accepted (2026-09-22)
- **Decision:** Use 0.4%, 0.8%, and 2.3% of horizon tokens; always include the full-cache oracle.
- **Consequence:** Track A mirrors large-model scarcity while retaining a true upper bound.

## D-006 — Equal tuning and a frozen test set

- **Status:** accepted (2026-09-22)
- **Decision:** Give every policy equal tuning effort and freeze and hash the test split before final evaluation.
- **Consequence:** The proposed policy cannot receive an unfair search advantage, and the final comparison remains auditable.
