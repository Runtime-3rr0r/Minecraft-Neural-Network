"""
sim/dataset_gen.py — Synthetic 5x5 Digit Dataset Generator

Extracted from: "Minecraft Redstone 5×5 Digit NN Optimizer - Grok.md"
and "Minecraft Digit Recognizer Pipeline & Optimizer Improvements - Claude.md"

Generates synthetic training data for the 5x5 digit recognizer neural network.
Digits 0-9 represented as 5x5 grids with procedural distortion pipeline.
No NumPy dependency — pure Python arithmetic as specified in the project.

Authorial constraint: weights 0-15, thresholds 0-375 (hidden) / 0-300 (output).
Pure Python stdlib only (random, math).
"""

import random
import math


# ========================== DIGIT TEMPLATES (5x5) ==========================
# Perfect clean versions — each digit as a list of 5 strings of length 5.
# These represent the "true" shape of each digit before distortion.
# All templates are centered in the 5x5 grid to match how a player actually draws.

DIGITS = {
    0: [
        "11111",
        "10001",
        "10001",
        "10001",
        "11111",
    ],
    1: [
        "00100",
        "01100",
        "00100",
        "00100",
        "01110",
    ],
    2: [
        "11110",
        "00010",
        "11110",
        "10000",
        "11111",
    ],
    3: [
        "11110",
        "00010",
        "11110",
        "00010",
        "11110",
    ],
    4: [
        "10010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    5: [
        "11111",
        "10000",
        "11110",
        "00010",
        "11110",
    ],
    6: [
        "11111",
        "10000",
        "11111",
        "10001",
        "11111",
    ],
    7: [
        "11111",
        "00010",
        "00100",
        "01000",
        "01000",
    ],
    8: [
        "11111",
        "10001",
        "11111",
        "10001",
        "11111",
    ],
    9: [
        "11111",
        "10001",
        "11111",
        "00001",
        "11111",
    ],
}


# ========================== CONVERSIONS ==========================

def to_array(grid):
    """Flatten a 5x5 grid of '0'/'1' strings into a list of 25 ints (0 or 1)."""
    return [int(c) for row in grid for c in row]


def one_hot(d):
    """Return a one-hot list of length 10 for digit d."""
    y = [0] * 10
    y[d] = 1
    return y


def one_hot_soft(d, noise=0.03):
    """Smoother labels for noisy handwriting — y[d] = 1.0, others = noise."""
    y = [noise] * 10
    y[d] = 1.0
    return y


# ========================== GEOMETRIC TRANSFORMATIONS ==========================

def shift(grid, dx, dy):
    """Shift a 5x5 grid by (dx, dy) pixels, zero-filling areas that move out."""
    new = [["0"] * 5 for _ in range(5)]
    for y in range(5):
        for x in range(5):
            nx, ny = x + dx, y + dy
            if 0 <= nx < 5 and 0 <= ny < 5:
                new[ny][nx] = grid[y][x]
    return ["".join(r) for r in new]


def random_shift(grid):
    """Apply a random shift with strength based on random choice."""
    strength = random.choice([0, 1, 1, 2])  # bias toward small shifts
    dx = random.randint(-strength, strength)
    dy = random.randint(-strength, strength)
    return shift(grid, dx, dy)


# ========================== STROKE-LEVEL DISTORTIONS ==========================

def flip_noise(grid, p=0.08):
    """Flip each pixel independently with probability p."""
    g = []
    for row in grid:
        nr = ""
        for c in row:
            if random.random() < p:
                nr += "1" if c == "0" else "0"
            else:
                nr += c
        g.append(nr)
    return g


def drop_pixels(grid, p=0.15):
    """Randomly drop (turn to 0) pixels with probability p."""
    g = []
    for row in grid:
        nr = ""
        for c in row:
            if c == "1" and random.random() < p:
                nr += "0"
            else:
                nr += c
        g.append(nr)
    return g


def add_noise(grid, p=0.08):
    """Randomly add (turn to 1) pixels with probability p."""
    g = []
    for row in grid:
        nr = ""
        for c in row:
            if c == "0" and random.random() < p:
                nr += "1"
            else:
                nr += c
        g.append(nr)
    return g


def erode(grid, p=0.15):
    """Erode: turn bordering 1s to 0s with probability p (structural thinning)."""
    g = [list(row) for row in grid]
    for y in range(5):
        for x in range(5):
            if grid[y][x] == "1":
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 5 and 0 <= ny < 5:
                        if random.random() < p:
                            g[y][x] = "0"
    return ["".join(row) for row in g]


def thicken(grid, p=0.25):
    """Thicken: turn bordering 0s to 1s with probability p (structural thickening)."""
    g = [list(row) for row in grid]
    for y in range(5):
        for x in range(5):
            if grid[y][x] == "1":
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 5 and 0 <= ny < 5:
                        if random.random() < p:
                            g[ny][nx] = "1"
    return ["".join(row) for row in g]


# ========================== MAIN DISTORTION PIPELINE ==========================

def distort(grid, severity=None):
    """
    Apply a sequence of distortions to a 5x5 digit grid.
    
    severity: float in [0, 1] controlling distortion intensity.
      - None: random severity ~ Uniform(0, 1)
      - If given: use exactly that severity value
    
    The pipeline applies distortions in order with probability governed by severity.
    Higher severity = more distortion applied.
    """
    if severity is None:
        severity = random.random()
    g = grid
    # spatial movement
    if severity > 0.2:
        g = random_shift(g)
    # structural distortions
    if severity > 0.4:
        g = flip_noise(g, 0.05 + severity * 0.05)
    if severity > 0.5:
        g = drop_pixels(g, 0.10 + severity * 0.10)
    if severity > 0.5:
        g = add_noise(g, 0.05)
    # morphological changes
    if severity > 0.4:
        g = erode(g, 0.12)
    if severity > 0.4:
        g = thicken(g, 0.20)
    return g


# ========================== DATASET GENERATION ==========================

def generate_dataset(samples_per_digit=100, use_soft_labels=False, severity=None):
    """
    Generate a synthetic dataset of 5x5 digit drawings.
    
    Args:
        samples_per_digit: number of distorted samples to generate per digit (0-9)
        use_soft_labels: if True, use soft labels (noise + one-hot); else hard one-hot
        severity: float in [0,1] or None for random per-sample severity
    
    Returns:
        list of (x, y) tuples where:
        - x: list of 25 ints (0/1) — the flattened 5x5 grid input
        - y: list of 10 ints (hard) or 10 floats (soft) — the target output
    """
    dataset = []
    for digit, grid in DIGITS.items():
        for _ in range(samples_per_digit):
            noisy = distort(grid, severity=severity)
            x = to_array(noisy)
            if use_soft_labels:
                y = one_hot_soft(digit)
            else:
                y = one_hot(digit)
            dataset.append((x, y))
    random.shuffle(dataset)
    return dataset


# ========================== DEBUG / TEST ==========================

if __name__ == "__main__":
    data = generate_dataset(samples_per_digit=10, use_soft_labels=True)
    print("Dataset size:", len(data))
    print("Example input:", data[0][0][:10], "...")  # first 10 of 25
    print("Example label:", data[0][1])