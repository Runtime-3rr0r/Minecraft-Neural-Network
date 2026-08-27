# Minecraft Neural Network

> **Status: 🟡 In progress · currently stale.** Built as a proof-of-concept, then
> set aside. **External help is very welcome** — see [CONTRIBUTING](#contributing).

A 5×5 digit recognizer designed as a neural network **inside Minecraft redstone**.
You draw a digit on a 5×5 grid; the machine works out which of 0–9 it is. No CPU,
no external compute — every multiply, add, and threshold is meant to live in the
game.

This repo is the **design record + a pure-Python simulator** of that idea. The
simulator is the working, testable thing; the redstone build is the goal.

## What it is

A small feed-forward network classifying hand-drawn digits:

```
5×5 input (25 pixels)
   ↓  binary 0/1
20 hidden neurons   (weighted sums + thresholds)
   ↓  integer weights 0–15
10 output neurons   (winner-take-all → digit 0–9)
```

Minecraft redstone only moves discrete signals, so the design is forced to use
**binary activations and integer weights** — no floats, no analog. That constraint
is the whole point: it makes a neural net run on adders, comparators, and RS
latches instead of a CPU.

## The docs (start here)

Rather than one wall of prose, the project is documented as focused papers —
each answers one piece of the design:

- **[docs/00-architecture.md](docs/00-architecture.md)** — the network: shape, neuron rule, exact math, parameter counts
- **[docs/01-dataset.md](docs/01-dataset.md)** — the procedural 5×5 digit generator and why synthetic
- **[docs/02-optimizer.md](docs/02-optimizer.md)** — training without backprop: simulated annealing over integer weights
- **[docs/03-redstone.md](docs/03-redstone.md)** — how it maps to redstone circuits, and what's built vs designed
- **[docs/04-status-and-roadmap.md](docs/04-status-and-roadmap.md)** — what works, what's stale, where help is wanted
- **[docs/05-code.md](docs/05-code.md)** — clean map of the `sim/` code + how it's sourced
- **[docs/05-provenance.md](docs/05-provenance.md)** — honest per-file provenance: what's literal from the sessions vs. reassembled
- **[docs/chats/](docs/chats/)** — the raw design sessions (ChatGPT / Grok / Claude transcripts) this all was mined from

## The code

A **pure-Python simulator** (no NumPy, integer math only) that mirrors the design
and trains the weights:

```
sim/
  dataset_gen.py   # synthetic 5x5 digit generator (with distortion modes)
  network.py       # binary forward pass, integer weights 0-15
  optimizer.py     # simulated annealing trainer
  classify.py      # CLI: train or classify a 5x5 grid
```

The simulator is the reference the redstone build should reproduce. See
[`docs/05-code.md`](docs/05-code.md) for a clean map of the modules and how each
mirrors the papers above.

## Project history / look-back

Useful for the "why is this shaped this way": the worked reasoning lives in
`docs/chats/` — the actual sessions (ChatGPT, Grok, Claude) that produced the
architecture, the dataset spec, the annealer, and the log-diagnostics along the
way. They're raw and chatty, but they're the honest record.

## Contributing

This is **open work, in progress, and currently stale** — I parked it and am
putting it out there for other people to build on, fix, or simply read.

- Findings and design docs are genuinely helpful as-is.
- Pull requests that push the **redstone build forward**, improve the simulator, or tighten the dataset are all welcome.
- Open an issue if you hit a contradiction in the docs — it's likely real as this is annotated by me, a human, not a reviewer.
- If you build any part of it in redstone, open a PR with a world download or schematic — that's the most valuable thing you could add.

## License

MIT — go build it.