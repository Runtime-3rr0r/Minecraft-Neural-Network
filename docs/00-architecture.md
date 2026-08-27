# 00 · Architecture

The network is a deliberately minimal, redstone-friendly feed-forward MLP.

## Shape

```
Input   25 pixels   (5×5 grid, binary 0/1)
Hidden  20 neurons
Output  10 neurons  (digit 0–9)
```

## Neuron rule

Every neuron is a **weighted-sum + threshold**, no bias:

```
output = 1  if  Σ(wᵢ · xᵢ) ≥ T
       = 0  otherwise
```

- `xᵢ ∈ {0,1}` — binary inputs/activations
- `wᵢ ∈ {0,…,15}` — integer weights (4-bit of precision)
- `T` — integer threshold, positive only (0–375 range observed)

## Why discrete

Redstone moves discrete signal strengths, so floats are out. Quantizing to
integer weights means the whole forward pass is:

- multiply = the weight (a static signal strength)
- sum = an adder chain
- threshold = a comparator against a reference

No multiplication circuit needed, because `(wᵢ·xᵢ)` is just `wᵢ` when the input
is 1, and 0 when it's 0. This is the core trick that makes a neural net
redstone-possible at all.

## No negatives

Redstone has no negative signal. The design uses **all-positive weights + a
threshold offset** to encode whatever sign a real weight would have — the
threshold swallows the bias.

## Parameter count

| Block | Count |
|------|-------|
| Input→hidden weights | 25 × 20 = 500 |
| Hidden→output weights | 20 × 10 = 200 |
| Hidden thresholds | 20 |
| Output thresholds | 10 |
| **Total** | **730** |

The `sim/` code reproduces this exactly.