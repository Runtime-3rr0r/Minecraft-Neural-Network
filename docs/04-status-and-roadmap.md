# 04 · Status & roadmap

## Status

**In progress · currently stale.** This is a proof-of-concept that was designed
and partly simulated, then set aside. Being honest about where it stands:

| Layer | State |
|-------|-------|
| Binary forward pass (algorithm) | ✅ working, testable |
| Pure-Python simulator (no NumPy) | ✅ working — trains and classifies |
| Synthetic dataset generator | ✅ working — balanced, distorted |
| Simulated-annealing optimizer | ✅ working (~76% accuracy) |
| Visualization tooling | ✅ network graphs, weight heatmaps |
| Single redstone neuron circuit | 🔧 designed, partial |
| Full 20-neuron redstone layer | 🔲 planned |
| 700-weight RS-latch memory | 🟠 designed, **not built** |
| In-game training interface | 🟠 designed, **not built** |

## Roadmap (where help is wanted)

**Immediate / high-value:**
- [ ] Take the single-neuron redstone design to a full hidden layer (the big one).
- [ ] Build the 700-weight RS-latch memory bank.
- [ ] Stand up the 5×5 draw input + display output in-game.

**Simulator / research:**
- [ ] Push accuracy past ~76% (bigger hidden layer? more annealing runs? better temp schedule?)
- [ ] Live drawing UI for the simulator with visual feedback.
- [ ] Reverse-engineer what each hidden neuron actually detects.

## What would unblock you as a contributor

- The `sim/` code is the ground truth the redstone should reproduce.
- `docs/chats/` has the full raw reasoning if a decision looks unmotivated.
- The stale spots are the redstone side — if you cactch the wiring flag, jump in.