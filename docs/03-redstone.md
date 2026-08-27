# 03 · Redstone implementation (the build goal)

How the network physically maps to redstone — and the honest state of that effort.

## The building blocks

Each hidden neuron reduces to three redstone pieces:

```
25 input wires  →  AND gates (gate each input × weight)  →  adder (sum)  →  comparator (≥ threshold)  →  1-bit output
```

- **Input layer:** 25 separate 1-bit wires (levers/buttons), one per pixel.
- **AND gating:** since `wᵢ·xᵢ` is `wᵢ` when the input is on and `0` when off, an
  AND gate per (input, weight) pair does the multiply.
- **Sum:** a large ripple-carry adder chain brings the gated weights together —
  25-input fan-in handled by chaining adders.
- **Threshold:** a comparator compares the accumulated sum against a reference
  signal; its output is the activation bit.
- **Weight storage:** RS-latch memory banks hold the 700 weights (500 in→hidden
  + 200 hidden→out), so weights can be changed in-game without rebuilding the
  wiring.

## The honest state

Be clear-eyed: **the redstone machine is a designed prototype, not a completed
500-column build.**

- ✅ Single-neuron building blocks (AND gating, 25-input adder, comparator) — **designed** and mapped from the Python.
- 🔧 Replicating that to a full 20-neuron hidden layer + 10-neuron output layer — **planned, not built**.
- 🔧 700-weight RS-latch memory bank + in-game training interface — **designed, not built**.

The screenshots in this repo are the redstone work-in-progress. The full build
is the open work-item — see the roadmap.

## Design notes (learned the hard way)

- **No negatives**: all-positive weights + threshold offset is the only sane way on redstone.
- **Large fan-in (25)**: don't use a single wide adder — chain ripple-carry stages.
- **Threshold tuning**: comparator reference signal is what you adjust.
- **Binary scaling**: fractional weights are represented with fixed-point integer arithmetic.