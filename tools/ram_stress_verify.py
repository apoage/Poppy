#!/usr/bin/env python3
"""RAM pattern stressor with verification — TASK_822 workstation stability probe.

Allocates N chunks of seeded random data, then loops: verify checksum of every
chunk, rewrite with a fresh pattern, log each pass with flush. Any mismatch =
memory corruption caught before a crash. Log survives hard resets.

Usage: python tools/ram_stress_verify.py --gb 18 --log runs/822_stability/ram_stress.log
"""
import argparse
import hashlib
import sys
import time

import numpy as np

CHUNK_MB = 512


def checksum(a: np.ndarray) -> str:
    return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=18.0)
    ap.add_argument("--log", default="runs/822_stability/ram_stress.log")
    args = ap.parse_args()

    n_chunks = int(args.gb * 1024 / CHUNK_MB)
    words = CHUNK_MB * 1024 * 1024 // 8
    log = open(args.log, "a", buffering=1)

    def say(msg: str) -> None:
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")

    say(f"START gb={args.gb} chunks={n_chunks} chunk_mb={CHUNK_MB}")
    rng = np.random.default_rng(822)
    chunks, sums = [], []
    for i in range(n_chunks):
        a = rng.integers(0, 2**63, size=words, dtype=np.int64)
        chunks.append(a)
        sums.append(checksum(a))
        if (i + 1) % 8 == 0:
            say(f"alloc {(i + 1) * CHUNK_MB // 1024}GB")

    errors = 0
    for p in range(1, 10**9):
        t0 = time.time()
        for i, a in enumerate(chunks):
            if checksum(a) != sums[i]:
                errors += 1
                say(f"!!! CORRUPTION pass={p} chunk={i} !!!")
            # rewrite with fresh pattern (exercises writes + keeps pages hot)
            a ^= np.int64(0x5A5A5A5A5A5A5A5A)
            sums[i] = checksum(a)
        say(f"pass={p} ok chunks={n_chunks} errors={errors} dt={time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
