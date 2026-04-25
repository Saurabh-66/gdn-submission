"""
Run this in a second terminal to monitor benchmark progress.
Usage:
    python watch_progress.py

It watches the Triton cache dir and the benchmark output for signs of life.
"""
import os
import time
import glob
import subprocess

CACHE_DIR = os.path.expanduser("~/.cache/flashinfer_bench/cache/triton")
CHECK_INTERVAL = 5  # seconds

def count_cached_kernels():
    return len(glob.glob(os.path.join(CACHE_DIR, "**/*.cubin"), recursive=True)) + \
           len(glob.glob(os.path.join(CACHE_DIR, "**/*.ptx"),   recursive=True))

def gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return out
    except Exception:
        return "nvidia-smi unavailable"

print("Watching benchmark progress. Ctrl+C to stop.")
print(f"Cache dir: {CACHE_DIR}")
print("-" * 60)

prev_kernels = 0
start = time.time()

while True:
    elapsed = time.time() - start
    n_kernels = count_cached_kernels()
    gpu = gpu_util()
    new = n_kernels - prev_kernels

    print(f"[{elapsed:6.0f}s] GPU: {gpu}%  |  cached kernels: {n_kernels} (+{new} new)")
    prev_kernels = n_kernels
    time.sleep(CHECK_INTERVAL)
