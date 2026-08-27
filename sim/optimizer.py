"""
sim/optimizer.py — Simulated Annealing / Genetic Algorithm Optimizer

Extracted from: "Minecraft Redstone 5×5 Digit NN Optimizer - Grok.md"
and "Log Analysis & Diagnostic - Grok.md" (bug fixes)

Network optimization for 25→20→10 binary threshold network.
- Weights: integers 0-15 only (4-bit)
- Thresholds: hidden 0-375, output 0-300
- Fitness: misclassification error (winner-take-all, argmax of raw sums)
- Search: Simulated annealing with adaptive mutation scales
- Special handling: 'dual signal' with invalid output penalties (N>=1 or N>1 outputs)
- Pure Python only (no NumPy) as required for embedded deployment

The optimizer evolved from pure genetic algorithm to include simulated annealing
for better exploration of the discrete search space.
"""

import random
import math
import copy
import time
from typing import List, Tuple, Dict, Any

# ========================== CONFIG ==========================
INPUT_SIZE = 25
HIDDEN_SIZE = 20
OUTPUT_SIZE = 10

W_MIN, W_MAX = 0, 15
HIDDEN_T_MIN, HIDDEN_T_MAX = 40, 220   # reasonable working range
OUTPUT_T_MIN, OUTPUT_T_MAX = 50, 250   # output thresholds

PENALTY_INVALID = 3          # penalty for all outputs same (0 active)
PENALTY_MULTIPLE = 1          # penalty for >1 active outputs
MUTATION_ELITE_FACTOR = 0.75   # probability of W1/W2 mutation for elites
MULTI_CANDIDATES = 4          # number of parallel candidates per step
STAGNATION_LIMIT = 300
SAMPLES_PER_DIGIT = 100

# ========================== INITIALIZATION ==========================

def init_network() -> Dict[str, Any]:
    """Initialize a random network with constrained parameters."""
    return {
        "W1": [[random.randint(W_MIN, W_MAX) for _ in range(HIDDEN_SIZE)]
               for _ in range(INPUT_SIZE)],
        "W2": [[random.randint(W_MIN, W_MAX) for _ in range(OUTPUT_SIZE)]
               for _ in range(HIDDEN_SIZE)],
        "T1": [random.randint(HIDDEN_T_MIN, HIDDEN_T_MAX) for _ in range(HIDDEN_SIZE)],
        "T2": [random.randint(OUTPUT_T_MIN, OUTPUT_T_MAX) for _ in range(OUTPUT_SIZE)],
    }


def stabilize(net: Dict[str, Any]) -> None:
    """Constrain thresholds to their valid ranges (in-place)."""
    for j in range(HIDDEN_SIZE):
        net["T1"][j] = max(HIDDEN_T_MIN, min(HIDDEN_T_MAX, net["T1"][j]))
    for k in range(OUTPUT_SIZE):
        net["T2"][k] = max(OUTPUT_T_MIN, min(OUTPUT_T_MAX, net["T2"][k]))


# ========================== FORWARD PASS ==========================

def forward(net: Dict[str, Any], x: List[int]) -> Tuple[List[int], List[int]]:
    """
    Exact redstone simulation for a single input.
    
    Returns:
        (binary_outputs, raw_output_sums)
    """
    W1, W2, T1, T2 = net["W1"], net["W2"], net["T1"], net["T2"]
    
    # Hidden layer: weighted sum of active inputs
    hidden_sums = [0] * HIDDEN_SIZE
    for j in range(HIDDEN_SIZE):
        s = 0
        for i in range(INPUT_SIZE):
            if x[i]:
                s += W1[i][j]
        hidden_sums[j] = s
    
    hidden = [1 if hidden_sums[j] >= T1[j] else 0 for j in range(HIDDEN_SIZE)]
    
    # Output layer: weighted sum of active hidden neurons
    out_sums = [0] * OUTPUT_SIZE
    for k in range(OUTPUT_SIZE):
        s = 0
        for j in range(HIDDEN_SIZE):
            if hidden[j]:
                s += W2[j][k]
        out_sums[k] = s
    
    outputs = [1 if out_sums[k] >= T2[k] else 0 for k in range(OUTPUT_SIZE)]
    
    return outputs, out_sums


# ========================== ERROR FUNCTION ==========================

def error(net: Dict[str, Any], dataset: List[Tuple[List[int], List[int]]]) -> int:
    """
    Compute classification error with dual-signal penalties.
    
    The error combines:
    1. Hamming distance between binary outputs and one-hot targets
    2. Penalties for invalid output patterns (all zeros or all ones)
    3. Strong penalty for wrong winner-take-all prediction
    """
    total = 0
    for x, y in dataset:
        outputs, out_sums = forward(net, x)
        
        # 1. Binary classification Hamming error
        for i in range(OUTPUT_SIZE):
            if outputs[i] != y[i]:
                total += 1
        
        # 2. Dual-signal penalties (invalid output patterns)
        active = sum(outputs)
        if active == 0:
            total += PENALTY_INVALID * 2
        elif active > 1:
            total += PENALTY_MULTIPLE
        
        # 3. Wrong prediction penalty (aligned with eventual winner-take-all use)
        predicted_digit = max(range(OUTPUT_SIZE), key=lambda k: out_sums[k])
        true_digit = max(range(OUTPUT_SIZE), key=lambda k: y[k])
        if predicted_digit != true_digit:
            total += 10  # strong penalty for wrong digit
    
    return total


# ========================== ADAPTIVE MUTATION ==========================

def mutate(net: Dict[str, Any], step: int, total_steps: int) -> Dict[str, Any]:
    """
    Adaptive mutation with scales that decrease over training.
    
    The mutation strength is proportional to (1 - progress), so early in
    training the network explores more broadly, later it fine-tunes.
    """
    n = copy.deepcopy(net)
    progress = step / total_steps
    scale = max(1, int(5 * (1 - progress)))  # big early, small late
    
    # Choose which layer to mutate
    choice = random.choice(["W1", "W2", "T1", "T2"])
    
    if choice == "W1":
        i = random.randint(0, INPUT_SIZE - 1)
        j = random.randint(0, HIDDEN_SIZE - 1)
        n["W1"][i][j] = max(W_MIN, min(W_MAX, n["W1"][i][j] + random.randint(-scale, scale)))
    elif choice == "W2":
        j = random.randint(0, HIDDEN_SIZE - 1)
        k = random.randint(0, OUTPUT_SIZE - 1)
        n["W2"][j][k] = max(W_MIN, min(W_MAX, n["W2"][j][k] + random.randint(-scale, scale)))
    elif choice == "T1":
        j = random.randint(0, HIDDEN_SIZE - 1)
        delta = random.randint(-10 * scale, 10 * scale)
        n["T1"][j] = max(HIDDEN_T_MIN, min(HIDDEN_T_MAX, n["T1"][j] + delta))
    elif choice == "T2":
        k = random.randint(0, OUTPUT_SIZE - 1)
        delta = random.randint(-10 * scale, 10 * scale)
        n["T2"][k] = max(OUTPUT_T_MIN, min(OUTPUT_T_MAX, n["T2"][k] + delta))
    
    stabilize(n)
    return n


# ========================== PRUNING ==========================

def prune(net: Dict[str, Any], threshold: int = 1) -> Dict[str, Any]:
    """Remove weak connections to compress the model."""
    for i in range(INPUT_SIZE):
        for j in range(HIDDEN_SIZE):
            if net["W1"][i][j] <= threshold:
                net["W1"][i][j] = 0
    for j in range(HIDDEN_SIZE):
        for k in range(OUTPUT_SIZE):
            if net["W2"][j][k] <= threshold:
                net["W2"][j][k] = 0
    return net


# ========================== TRAINING LOOP (SIMULATED ANNEALING) ==========================

def train(dataset: List[Tuple[List[int], List[int]]], steps: int = 2000) -> Dict[str, Any]:
    """
    Simulated annealing trainer with parallel candidate search.
    
    At each step:
    1. Generate MULTI_CANDIDATES mutated variants of current network
    2. Evaluate all candidates in parallel
    3. Accept new network based on error improvement AND temperature schedule
    """
    net = init_network()
    best = copy.deepcopy(net)
    best_err = error(net, dataset)
    
    temp = 5.0  # starting temperature
    start_time = time.time()
    avg_step = 0.01
    alpha = 0.02  # smoothing factor for ETA calculation
    
    print(f"Starting training for {steps} steps...")
    print(f"Dataset: {len(dataset)} samples ({len(dataset)//10} per digit)")
    
    for i in range(steps):
        step_start = time.time()
        
        # --- Multi-candidate parallel search ---
        best_cand = None
        best_cand_err = float('inf')
        
        for _ in range(MULTI_CANDIDATES):
            cand = mutate(net, i, steps)
            e = error(cand, dataset)
            if e < best_cand_err:
                best_cand = cand
                best_cand_err = e
        
        # --- Simulated annealing acceptance ---
        delta = best_cand_err - best_err
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.0001)):
            net = best_cand
            best_err = best_cand_err
            best = copy.deepcopy(cand)
            stabilize(net)
        
        temp *= 0.999  # cooling schedule
        
        # --- ETA tracking ---
        step_time = time.time() - step_start
        avg_step = (1 - alpha) * avg_step + alpha * step_time
        
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - start_time
            eta = (steps - i) * avg_step / 60  # minutes remaining
            print(
                f"step {i:4d} | "
                f"err {best_err:4d} | "
                f"temp {temp:.3f} | "
                f"ETA {eta:.2f} min"
            )
    
    print(f"Training complete. Best error: {best_err}/{len(dataset)}")
    return best


# ========================== BACKWARD COMPATIBILITY LAYER ==========================

def load_model_legacy(path: str = "trained_model_v2.json") -> Dict[str, Any]:
    """Load a saved model from the older JSON format."""
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    # Convert lists back to nested structure
    net = {
        "W1": [[data["W1"][i][j] for j in range(HIDDEN_SIZE)] for i in range(INPUT_SIZE)],
        "W2": [[data["W2"][j][k] for k in range(OUTPUT_SIZE)] for j in range(HIDDEN_SIZE)],
        "T1": data["T1"],
        "T2": data["T2"],
    }
    return net


def save_model_legacy(net: Dict[str, Any], path: str = "trained_model_v2.json") -> None:
    """Save model to legacy JSON format for compatibility."""
    import json
    # Convert to flat representation for JSON
    data = {
        "W1": [[net["W1"][i][j] for j in range(HIDDEN_SIZE)] for i in range(INPUT_SIZE)],
        "W2": [[net["W2"][j][k] for k in range(OUTPUT_SIZE)] for j in range(HIDDEN_SIZE)],
        "T1": net["T1"],
        "T2": net["T2"],
    }
    with open(path, 'w') as f:
        json.dump(data, f)


# ========================== DEMO & TESTING ==========================

if __name__ == "__main__":
    # Import dataset generator (relative import)
    from dataset_gen import generate_dataset
    
    print("="*60)
    print("MINECRAFT NN SIMULATOR — OPTIMIZER DEMO")
    print("="*60)
    
    # Generate training data
    dataset = generate_dataset(
        samples_per_digit=SAMPLES_PER_DIGIT,
        use_soft_labels=True
    )
    
    print(f"Generated {len(dataset)} training samples")
    print("Starting simulated annealing training...")
    
    # Train the network
    model = train(dataset, steps=2000)
    
    # Prune weak connections
    model = prune(model)
    
    # Save the trained model
    save_model_legacy(model)
    print("Saved model to trained_model_v2.json")
    
    # Print redstone-ready configuration
    print("\n" + "="*60)
    print("REDSRONE-WREADY CONFIGURATION")
    print("="*60)
    print("INPUT → HIDDEN LAYER (25 weights + threshold per hidden neuron):")
    for j in range(HIDDEN_SIZE):
        weights = [model["W1"][i][j] for i in range(INPUT_SIZE)]
        print(f"  Hidden_{j:02d}: weights={weights}, threshold={model['T1'][j]}")
    
    print("\nHIDDEN → OUTPUT LAYER (20 weights + threshold per output neuron):")
    for k in range(OUTPUT_SIZE):
        weights = [model["W2"][j][k] for j in range(HIDDEN_SIZE)]
        print(f"  Output_{k:02d} (digit {k}): weights={weights}, threshold={model['T2'][k]}")
    
    print("\nCopy the lists above directly into your redstone build.")
    print("At inference: compute the 10 output sums and pick the highest (winner-take-all).")
    print("="*60)