"""
sim/network.py — Binary/Quantized Feed-Forward Network Definition

Extracted from: "Binary Neural Network for Minecraft Redstone - Grok.md"
and "Minecraft Neural Network Optimizer Fixes & Dual-Signal Implementation - ChatGPT.md"

Network architecture: 25 inputs -> 20 hidden -> 10 outputs
- Weights: integers 0-15 only (4-bit)
- Thresholds: per-neuron integers (hidden: 0-375, output: 0-300)
- Forward pass: sum of weights for active inputs (where x_i == 1), compare to threshold
- Output: binary (1 if sum >= threshold else 0)
- Winner-take-all classification: argmax of raw output sums
- No bias (per design recommendation — folded into threshold)
- Pure Python arithmetic (no NumPy)

This is the EXACT redstone computation model.
"""

# ========================== NETWORK CONFIG ==========================
N_IN = 25
N_HID = 20
N_OUT = 10

W_MAX = 15
T1_MAX = 375   # hidden layer max sum = 25 * 15
T2_MAX = 300   # output layer max sum = 20 * 15

NUM_W1 = N_IN * N_HID          # 500
NUM_W2 = N_HID * N_OUT         # 200
NUM_T1 = N_HID                 # 20
NUM_T2 = N_OUT                 # 10
GENOME_SIZE = NUM_W1 + NUM_W2 + NUM_T1 + NUM_T2  # 730


# ========================== GENOME <-> NETWORK ==========================

def decode_genome(genome):
    """
    Decode flat genome array into network components with constraint clipping.
    
    Args:
        genome: list/array of ints, length GENOME_SIZE (730)
    
    Returns:
        (w1, w2, th1, th2) where:
        - w1: list of 25 lists of 20 ints (INPUT x HIDDEN)
        - w2: list of 20 lists of 10 ints (HIDDEN x OUTPUT)
        - th1: list of 20 ints (hidden thresholds)
        - th2: list of 10 ints (output thresholds)
    """
    # Input -> Hidden weights
    w1 = [genome[i * N_HID:(i + 1) * N_HID] for i in range(N_IN)]
    
    # Hidden -> Output weights
    w2_start = NUM_W1
    w2 = [genome[w2_start + j * N_OUT:w2_start + (j + 1) * N_OUT] for j in range(N_HID)]
    
    # Hidden thresholds
    th1_start = NUM_W1 + NUM_W2
    th1 = genome[th1_start:th1_start + NUM_T1]
    
    # Output thresholds
    th2_start = th1_start + NUM_T1
    th2 = genome[th2_start:th2_start + NUM_T2]
    
    # Enforce exact constraints
    for i in range(N_IN):
        for j in range(N_HID):
            w1[i][j] = max(0, min(W_MAX, w1[i][j]))
    for j in range(N_HID):
        for k in range(N_OUT):
            w2[j][k] = max(0, min(W_MAX, w2[j][k]))
    for j in range(N_HID):
        th1[j] = max(0, min(T1_MAX, th1[j]))
    for k in range(N_OUT):
        th2[k] = max(0, min(T2_MAX, th2[k]))
    
    return w1, w2, th1, th2


def encode_genome(w1, w2, th1, th2):
    """Encode network components back into a flat genome array."""
    genome = []
    for i in range(N_IN):
        genome.extend(w1[i])
    for j in range(N_HID):
        genome.extend(w2[j])
    genome.extend(th1)
    genome.extend(th2)
    return genome


def init_random_genome():
    """Create a fresh random genome respecting all constraints."""
    import random
    genome = []
    # W1: 0-15 uniform
    genome.extend([random.randint(0, W_MAX) for _ in range(NUM_W1)])
    # W2: 0-15 uniform
    genome.extend([random.randint(0, W_MAX) for _ in range(NUM_W2)])
    # T1: 0-375, but biased toward middle range for initialization
    genome.extend([random.randint(80, 180) for _ in range(NUM_T1)])
    # T2: 0-300, similarly biased
    genome.extend([random.randint(50, 200) for _ in range(NUM_T2)])
    return genome


# ========================== FORWARD PASS (EXACT REDSTONE MATCH) ==========================

def forward(inputs, w1, w2, th1, th2):
    """
    Exact forward pass matching the Minecraft redstone computation.
    
    Args:
        inputs: list of 25 ints (0 or 1) — the 5x5 grid flattened
        w1: INPUT x HIDDEN weight matrix (25 x 20), values 0-15
        w2: HIDDEN x OUTPUT weight matrix (20 x 10), values 0-15
        th1: list of 20 ints — hidden layer thresholds (0-375)
        th2: list of 10 ints — output layer thresholds (0-300)
    
    Returns:
        (out_act, out_sums) where:
        - out_act: list of 10 ints (0/1) — binary output activations
        - out_sums: list of 10 ints — raw sums before thresholding
    
    The computation:
      1. For each hidden neuron j: sum(w1[i][j] for i where inputs[i] == 1)
         Compare sum to th1[j] → binary hidden activation
      2. For each output neuron k: sum(w2[j][k] for j where hidden[j] == 1)
         Compare sum to th2[k] → binary output activation
    """
    # Hidden layer: weighted sum of active inputs
    hid_sums = [0] * N_HID
    for j in range(N_HID):
        s = 0
        for i in range(N_IN):
            if inputs[i]:
                s += w1[i][j]
        hid_sums[j] = s
    
    # Binary hidden activations
    hid_act = [1 if hid_sums[j] >= th1[j] else 0 for j in range(N_HID)]
    
    # Output layer: weighted sum of active hidden neurons
    out_sums = [0] * N_OUT
    for k in range(N_OUT):
        s = 0
        for j in range(N_HID):
            if hid_act[j]:
                s += w2[j][k]
        out_sums[k] = s
    
    # Binary output activations
    out_act = [1 if out_sums[k] >= th2[k] else 0 for k in range(N_OUT)]
    
    return out_act, out_sums


def predict(inputs, w1, w2, th1, th2):
    """
    Winner-take-all prediction on raw output sums (as used in Minecraft at inference).
    
    Args:
        inputs: list of 25 ints (0 or 1)
        w1, w2, th1, th2: network parameters
    
    Returns:
        int 0-9 — the predicted digit (argmax of out_sums)
    """
    _, out_sums = forward(inputs, w1, w2, th1, th2)
    # Winner-take-all: pick digit with highest raw sum
    return max(range(N_OUT), key=lambda k: out_sums[k])


def forward_batch(inputs_batch, w1, w2, th1, th2):
    """
    Batch forward pass for multiple inputs (pure Python, no NumPy).
    
    Args:
        inputs_batch: list of input vectors (each is list of 25 ints)
        w1, w2, th1, th2: network parameters
    
    Returns:
        list of (out_act, out_sums) for each input
    """
    return [forward(x, w1, w2, th1, th2) for x in inputs_batch]


# ========================== WEIGHT PRUNING (POST-TRAINING) ==========================

def prune_weights(w1, w2, threshold=1):
    """
    Remove useless weak connections after training.
    Mutates w1 and w2 in place, setting weights <= threshold to 0.
    """
    for i in range(N_IN):
        for j in range(N_HID):
            if w1[i][j] <= threshold:
                w1[i][j] = 0
    for j in range(N_HID):
        for k in range(N_OUT):
            if w2[j][k] <= threshold:
                w2[j][k] = 0


# ========================== REDSTONE OUTPUT FORMATTING ==========================

def print_for_redstone(w1, w2, th1, th2):
    """Print weights and thresholds in format ready for Minecraft redstone build."""
    print("INPUT → HIDDEN LAYER (25 weights + threshold per hidden neuron):")
    for j in range(N_HID):
        weights = [w1[i][j] for i in range(N_IN)]
        print(f"  Hidden_{j:02d}: weights={weights}, threshold={th1[j]}")
    
    print("\nHIDDEN → OUTPUT LAYER (20 weights + threshold per output neuron):")
    for k in range(N_OUT):
        weights = [w2[j][k] for j in range(N_HID)]
        print(f"  Output_{k:02d} (digit {k}): weights={weights}, threshold={th2[k]}")
    
    print("\nCopy the lists above directly into your redstone build.")
    print("At inference: compute the 10 output sums and pick the highest (winner-take-all).")


if __name__ == "__main__":
    # Quick self-test
    genome = init_random_genome()
    w1, w2, th1, th2 = decode_genome(genome)
    
    # Test input: digit 0 pattern
    test_input = [
        1,1,1,1,1,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        1,1,1,1,1,
    ]
    
    out_act, out_sums = forward(test_input, w1, w2, th1, th2)
    pred = predict(test_input, w1, w2, th1, th2)
    
    print("Self-test:")
    print(f"  Hidden sums: {sum(hid_act) if (hid_act := [1 if sum(w1[i][j] for i in range(N_IN) if test_input[i]) >= th1[j] else 0 for j in range(N_HID)]) else 'N/A'} active")
    print(f"  Output sums: {out_sums}")
    print(f"  Binary out:  {out_act}")
    print(f"  Prediction:  {pred}")
    print(f"\nGenome size: {len(genome)} (expected {GENOME_SIZE})")