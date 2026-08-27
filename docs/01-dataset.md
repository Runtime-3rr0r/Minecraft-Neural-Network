# 01 · The dataset

Real handwriting is messy. A fixed set of a few clean 5×5 templates would let
the network memorize the templates instead of learning the *idea* of a digit, so
the project uses a **procedural synthetic generator**.

## Approach

- Start from centered 5×5 templates for each digit from a small hand-built seed set.
- Apply **distortion modes** to make the samples more realistic:

  | Mode | Shift | Noise |
  |------|-------|-------|
  | Light | ±1 px | minor |
  | Medium | ±2 px | occasional pixel flips |
  | Heavy | ±3 px | significant flips / missing pixels |

- A "clean ratio" (≈60% after fixes) keeps a floor of legible examples so the
  network still sees good digits.

## Why procedural

- Matches the messy, imperfect way a person actually draws on a 5×5 grid.
- Gives **unbounded sample variety** instead of a tiny fixed set.
- Avoids overfitting to a handful of perfect templates.

## Target

Balanced 10-class coverage (all digits 0–9 roughly equal), which the generator
enforces so the optimizer isn't just memorizing the common ones.

The generator lives in `sim/dataset_gen.py`; run it standalone to inspect the
distorted samples it produces.