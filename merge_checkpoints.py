#!/usr/bin/env python3
"""
Merge two Easy Mode checkpoints via weight averaging.

Usage:
    python merge_checkpoints.py jax_ckpt_easy_A/phase1_best.pkl jax_ckpt_easy_B/phase1_best.pkl -o merged.pkl
"""

import argparse
import pickle
from pathlib import Path
import jax.numpy as jnp
import jax.tree_util as jtu


def average_params(params_a, params_b, alpha=0.5):
    """Average two parameter trees: alpha * A + (1-alpha) * B"""
    return jtu.tree_map(
        lambda a, b: alpha * a + (1 - alpha) * b,
        params_a, params_b
    )


def load_checkpoint(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(ckpt, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(ckpt, f)


def main():
    parser = argparse.ArgumentParser(description='Merge two checkpoints')
    parser.add_argument('ckpt_a', type=str, help='First checkpoint')
    parser.add_argument('ckpt_b', type=str, help='Second checkpoint')
    parser.add_argument('-o', '--output', type=str, default='merged.pkl', help='Output path')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for A (default 0.5 = equal)')
    args = parser.parse_args()

    print(f"Loading {args.ckpt_a}...")
    ckpt_a = load_checkpoint(args.ckpt_a)

    print(f"Loading {args.ckpt_b}...")
    ckpt_b = load_checkpoint(args.ckpt_b)

    # Merge params
    print(f"Merging with alpha={args.alpha} (A={args.alpha:.0%}, B={1-args.alpha:.0%})...")
    merged_params = average_params(ckpt_a['params'], ckpt_b['params'], args.alpha)

    # Use A's structure, replace params
    merged_ckpt = {
        'params': merged_params,
        'epoch': max(ckpt_a.get('epoch', 0), ckpt_b.get('epoch', 0)),
        'metrics': {
            'source': 'merged',
            'alpha': args.alpha,
            'a_agreement': ckpt_a.get('metrics', {}).get('oracle_agreement', 0),
            'b_agreement': ckpt_b.get('metrics', {}).get('oracle_agreement', 0),
        }
    }

    # Copy optimizer state from better model
    a_agree = ckpt_a.get('metrics', {}).get('oracle_agreement', 0)
    b_agree = ckpt_b.get('metrics', {}).get('oracle_agreement', 0)
    better = ckpt_a if a_agree >= b_agree else ckpt_b
    if 'opt_state' in better:
        merged_ckpt['opt_state'] = better['opt_state']

    print(f"Saving to {args.output}...")
    save_checkpoint(merged_ckpt, args.output)

    print(f"Done! A={a_agree:.1%}, B={b_agree:.1%} → Merged")


if __name__ == '__main__':
    main()
