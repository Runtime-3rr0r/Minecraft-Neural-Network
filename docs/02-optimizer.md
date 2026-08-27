# 02 · Optimizer (training without backprop)

There are no gradients here — the weights are discrete integers on a
non-differentiable landscape, so gradient descent can't run. Training instead
uses **simulated annealing**, a temperature-controlled random search.

## The optimizer

- Searches over the 730 integer parameters.
- At each step, proposes a small random weight perturbation and measures the change in error `ΔE`.
- Accepts the move if it improves the error, **or** with probability `exp(−ΔE / T)` if it's worse.
- Temperature `T` starts high (lots of exploration) and cools slowly, so the
  search first ranges widely, then locks into a good well.

This is a natural fit: the weight space is discrete and rugged, and annealing
doesn't need a gradient.

## Accuracy

The simulator's annealer reached **~76% test accuracy** on the synthetic set —
up from ~45% after removing a redundant term in the optimizer objective. Not a
benchmark-beater, but it proves the pipeline and the annealer genuinely work
on a 10-class task with only 25 binary inputs.

## Training pipeline

1. **Offline (Python)** — generate a large balanced set, anneal the weights.
2. **Manual entry** — copy the resulting integer weights into redstone memory
   (RS-latch banks) or the simulator's weight table.
3. **Validate** — classify held-out/noised digits.
4. **Fine-tune** — small hill-climbing nudges where a digit misbehaves.

The trainer is `sim/optimizer.py`; `sim/classify.py` runs the full loop.