#!/usr/bin/env python3
"""Compare smoke test A/B/C results side-by-side."""

import csv
import glob
import sys
from pathlib import Path


def load_csv(pattern):
    """Load latest CSV matching pattern, return list of dicts."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None
    path = files[-1]
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return path, rows


def fmt(val, width=8):
    try:
        return f"{float(val):>{width}.4f}"
    except (ValueError, TypeError):
        return f"{'N/A':>{width}}"


def main():
    variants = {
        "A (ctrl)": "logs/smoke_A_control*.csv",
        "B (h2048)": "logs/smoke_B_hidden2048*.csv",
        "C (noAR)": "logs/smoke_C_hidden2048_noAR*.csv",
    }

    data = {}
    for label, pattern in variants.items():
        path, rows = load_csv(pattern)
        if rows:
            data[label] = rows
            print(f"  {label}: {path} ({len(rows)} epochs)")
        else:
            print(f"  {label}: NO DATA")

    if not data:
        print("\nNo CSVs found. Are the runs still going?")
        sys.exit(1)

    # Key metrics
    metrics = ["agree_trade", "agree_hold", "oracle_agreement", "pred_hold_ratio", "total_loss", "grad_norm"]

    print("\n" + "=" * 80)
    print("SMOKE TEST COMPARISON")
    print("=" * 80)

    # Header
    header = f"{'epoch':>6}"
    for label in data:
        header += f" | {'agree_trade':>11} {'agree_hold':>11} {'oracle_agr':>11} {'pred_hold':>11}"
    print(f"\n{header}")
    print(f"{'':>6}", end="")
    for label in data:
        print(f" | {label:^47}", end="")
    print()
    print("-" * (7 + 50 * len(data)))

    # Find common epochs
    max_epochs = max(len(rows) for rows in data.values())
    for i in range(max_epochs):
        epoch_str = ""
        for label, rows in data.items():
            if i < len(rows):
                row = rows[i]
                if not epoch_str:
                    epoch_str = f"{row.get('epoch', i):>6}"
                epoch_str += f" | {fmt(row.get('agree_trade'), 11)} {fmt(row.get('agree_hold'), 11)} {fmt(row.get('oracle_agreement'), 11)} {fmt(row.get('pred_hold_ratio'), 11)}"
            else:
                epoch_str += f" | {'—':>11} {'—':>11} {'—':>11} {'—':>11}"
        if epoch_str:
            print(epoch_str)

    # Summary: last 5 epochs average
    print("\n" + "=" * 80)
    print("LAST 5 EPOCH AVERAGES")
    print("=" * 80)
    print(f"{'Variant':<12} {'agree_trade':>12} {'agree_hold':>12} {'oracle_agr':>12} {'pred_hold':>12} {'total_loss':>12}")
    print("-" * 72)

    for label, rows in data.items():
        tail = rows[-5:] if len(rows) >= 5 else rows
        avgs = {}
        for m in ["agree_trade", "agree_hold", "oracle_agreement", "pred_hold_ratio", "total_loss"]:
            vals = [float(r[m]) for r in tail if m in r]
            avgs[m] = sum(vals) / len(vals) if vals else 0
        print(f"{label:<12} {avgs['agree_trade']:>12.4f} {avgs['agree_hold']:>12.4f} {avgs['oracle_agreement']:>12.4f} {avgs['pred_hold_ratio']:>12.4f} {avgs['total_loss']:>12.4f}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    labels = list(data.keys())
    if len(labels) >= 2:
        def avg_trade(label):
            rows = data[label]
            tail = rows[-5:] if len(rows) >= 5 else rows
            vals = [float(r["agree_trade"]) for r in tail if "agree_trade" in r]
            return sum(vals) / len(vals) if vals else 0

        scores = {l: avg_trade(l) for l in labels}
        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)

        print(f"  Best agree_trade:  {best} ({scores[best]:.4f})")
        print(f"  Worst agree_trade: {worst} ({scores[worst]:.4f})")

        if "A (ctrl)" in scores and "B (h2048)" in scores:
            delta = scores["B (h2048)"] - scores["A (ctrl)"]
            print(f"\n  B vs A (hidden_dim effect): {'+' if delta > 0 else ''}{delta:.4f}")
            if delta > 0.02:
                print("  -> CONFIRMED: hidden_dim=2048 improves trade learning")
            elif delta < -0.02:
                print("  -> UNEXPECTED: hidden_dim=2048 hurts (investigate)")
            else:
                print("  -> INCONCLUSIVE: difference too small")

        if "B (h2048)" in scores and "C (noAR)" in scores:
            delta = scores["C (noAR)"] - scores["B (h2048)"]
            print(f"\n  C vs B (autoregressive effect): {'+' if delta > 0 else ''}{delta:.4f}")
            if delta > 0.02:
                print("  -> CONFIRMED: model-state rollout hurts trade learning")
            elif delta < -0.02:
                print("  -> UNEXPECTED: model-state rollout helps (investigate)")
            else:
                print("  -> INCONCLUSIVE: difference too small")


if __name__ == "__main__":
    main()
