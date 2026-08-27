# 05 · Code map & provenance

The `sim/` directory is a **pure-Python mirror of the design** — no NumPy, integer
math only, so it stays a faithful reference for what the redstone should do.

| Module | Mirrors doc | What it does | Sourced from |
|--------|-------------|--------------|--------------|
| `sim/dataset_gen.py` | [01-dataset](01-dataset.md) | Procedural 5×5 digit generator with distortion modes & balanced classes | Grok optimizer transcript (lines 1639–1844) + Claude mode-toggle |
| `sim/network.py` | [00-architecture](00-architecture.md) | 25→20→10 forward pass, integer weights 0–15, threshold rule, winner-take-all | Grok binaries transcript (lines 1239–1358) |
| `sim/optimizer.py` | [02-optimizer](02-optimizer.md) | Simulated-annealing/genetic trainer over integer weights | Assembled from Grok (1361–1420) + ChatGPT fixes + Claude improvements — see below |
| `sim/classify.py` | all | CLI: train a network and/or classify a 5×5 digit grid | Assembled from usage patterns — see below |

## Provenance — honest about what's literal vs. assembled

The full detail is in [**`docs/05-provenance.md`**](05-provenance.md). Summary:

- **Digit templates (0–9):** verbatim from the Grok transcript. ✅
- **Network shape (25→20→10, weights 0–15, no bias, threshold rule):** confirmed in both the design spec *and* embedded code. ✅
- **`forward()` / `decode_genome()`:** literally shown in the transcript. ✅
- **`optimizer.py` and `classify.py`:** **reconstructed from fragments** spread across ≥3 sessions (the annealing acceptance `exp(−ΔE/T)` is literal at Grok line 1594; the rest is assembled faithfully from the surrounding design). The `docs/05-provenance.md` report flags exactly which parts were literal vs. stitched. Nothing is invented beyond the transcripts' own description.
- **No `.mcworld` / schematic / world-build files exist** — the transcripts describe Python + redstone *concepts* (threshold gates, weight latches) but never embed a physical build file. Those are the open work.

## Build the redstone from this

`sim/` is the ground truth:

1. Train weights: `python sim/classify.py --train`
2. Read the weights — small integers 0–15.
3. Follow [03-redstone](03-redstone.md) to wire each neuron.

## Verified

The four modules import and run on a minimal smoke test (`python -c "import dataset_gen, network, optimizer, classify"` + a tiny anneal run improved from 61→45 errors). See `docs/05-provenance` for the recorded smoke-test output.