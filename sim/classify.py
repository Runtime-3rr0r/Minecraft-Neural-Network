"""
sim/classify.py — CLI for Training and Classifying 5x5 Digit Grids

Assembled from: patterns in "Minecraft Digit Recognizer Pipeline & Optimizer Improvements - Calude.md"
and "Minecraft Redstone 5×5 Digit NN Optimizer - Grok.md"

Usage:
    python classify.py --train          # train a new model
    python classify.py --classify 1,1,0,...  # classify a 5x5 grid

The CLI allows both training (simulated annealing) and inference
(winner-take-all classification) using the pure-Python network.
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network import init_random_genome, decode_genome, forward, predict
from optimizer import init_network, train, error, prune, save_model_legacy
from dataset_gen import generate_dataset, to_array, DIGITS


def parse_grid_arg(text: str) -> list:
    """Parse a 5x5 grid from command line (comma-separated or space-separated)."""
    # Handle both "0,1,0,0,1" and "01101" formats
    # Remove any commas, then split on whitespace
    text = text.replace(",", " ")
    # Extract only 0/1 characters
    digits = []
    for c in text:
        if c in "01":
            digits.append(int(c))
    if len(digits) != 25:
        raise ValueError(f"Expected 25 binary values for 5x5 grid, got {len(digits)}")
    return digits


def print_grid(grid: list):
    """Pretty-print a 5x5 grid as 0/1 strings."""
    for row in range(5):
        print(" ".join(str(grid[row*5 + i]) for i in range(5)))


def main():
    parser = argparse.ArgumentParser(
        description="Minecraft 5x5 Digit NN — Train and Classify CLI"
    )
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument(
        "--classify", type=str, metavar="GRID",
        help="Classify a 25-digit binary grid (e.g., '0,1,1,0,0,...' or '01100...')"
    )
    parser.add_argument(
        "--model", type=str, default="trained_model_v2.json",
        help="Path to saved model (default: trained_model_v2.json)"
    )
    parser.add_argument(
        "--steps", type=int, default=2000,
        help="Training steps (default: 2000)"
    )
    parser.add_argument(
        "--dataset-size", type=int, default=100,
        help="Samples per digit (default: 100)"
    )
    args = parser.parse_args()
    
    if args.train:
        print("="*60)
        print("TRAINING MODE")
        print("="*60)
        dataset = generate_dataset(samples_per_digit=args.dataset_size, use_soft_labels=True)
        print(f"Generated dataset: {len(dataset)} samples")
        model = train(dataset, steps=args.steps)
        model = prune(model)
        save_model_legacy(model, path=args.model)
        print(f"Model saved to: {args.model}")
        print("="*60)
    
    if args.classify:
        # Load or use fresh model
        if os.path.exists(args.model):
            import json
            with open(args.model) as f:
                data = json.load(f)
            # Rebuild network from JSON
            net = {
                "W1": [[data["W1"][i][j] for j in range(20)] for i in range(25)],
                "W2": [[data["W2"][j][k] for k in range(10)] for j in range(20)],
                "T1": data["T1"],
                "T2": data["T2"],
            }
            print(f"Loaded model from {args.model}")
        else:
            print("No saved model found — using random initialization (for demo only)")
            net = init_network()
        
        grid = parse_grid_arg(args.classify)
        print("\nInput grid (5x5):")
        print_grid(grid)
        
        pred = predict(grid, net["W1"], net["W2"], net["T1"], net["T2"])
        
        _, out_sums = forward(grid, net["W1"], net["W2"], net["T1"], net["T2"])
        
        print(f"\nOutput sums: {out_sums}")
        print(f"Predicted digit: {pred}")
        
        # Show which digit template matches closest
        best_template_digit = None
        best_template_score = -1
        for d, template_strs in DIGITS.items():
            template_flat = [int(c) for row in template_strs for c in row]
            score = sum(1 for a, b in zip(grid, template_flat) if a == b)
            if score > best_template_score:
                best_template_score = score
                best_template_digit = d
        
        print(f"Closest template digit (Hamming): {best_template_digit} ({best_template_score}/25 matches)")
    
    if not args.train and not args.classify:
        parser.print_help()


if __name__ == "__main__":
    import os
    main()