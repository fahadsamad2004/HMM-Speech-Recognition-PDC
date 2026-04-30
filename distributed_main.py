"""
distributed_main.py - Milestone 2: Distributed HMM Training Entry Point

Runs the complete scalability experiment:
  - Tests 1, 2, 4, 8 workers on same dataset
  - Measures speedup, efficiency, communication overhead
  - Verifies distributed results match sequential baseline
  - Runs full distributed training on all phoneme classes

Run:
    python distributed_main.py
"""

import os
import sys
import time
import numpy as np
import ray
from collections import Counter, defaultdict

from distributed_trainer import DistributedHMMTrainer, DistributedSpeechRecognizer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TIMIT_ROOT    = r"C:\Users\fahad\OneDrive\Documents\IBA\spring26\PDC\HMM_Project\timit"
N_MFCC        = 13
N_STATES      = 5
N_ITER        = 15

# Worker counts to benchmark — your machine has 8 logical processors
WORKER_COUNTS = [1, 2, 4, 8]

# Subset size for scalability benchmark (keeps benchmark fast)
# Uses a balanced subset across all phoneme classes
SUBSET_SIZE   = None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(features):
    mean = features.mean(axis=0)
    std  = features.std(axis=0) + 1e-8
    return (features - mean) / std


def load_timit(timit_root, n_mfcc=13):
    import librosa
    train_data, test_data = [], []
    skipped = 0
    SR = 16000

    for split in ['TRAIN', 'TEST']:
        split_dir = os.path.join(timit_root, split)
        if not os.path.exists(split_dir):
            print(f"  Warning: {split_dir} not found.")
            continue
        print(f"  Loading {split}...")
        count = 0
        for dr in sorted(os.listdir(split_dir)):
            dr_dir = os.path.join(split_dir, dr)
            if not os.path.isdir(dr_dir):
                continue
            for speaker in sorted(os.listdir(dr_dir)):
                spk_dir = os.path.join(dr_dir, speaker)
                if not os.path.isdir(spk_dir):
                    continue
                for fname in sorted(os.listdir(spk_dir)):
                    if not fname.upper().endswith('.WAV'):
                        continue
                    wav  = os.path.join(spk_dir, fname)
                    phn  = os.path.join(spk_dir, fname.upper().replace('.WAV','.PHN'))
                    if not os.path.exists(phn):
                        phn = os.path.join(spk_dir, fname.lower().replace('.wav','.phn'))
                    if not os.path.exists(phn):
                        skipped += 1; continue
                    try:
                        y, _ = librosa.load(wav, sr=SR)
                        with open(phn) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) != 3: continue
                                s, e, label = int(parts[0]), int(parts[1]), parts[2]
                                if label == 'h#': continue
                                seg = y[s:e]
                                if len(seg) < SR * 0.02: continue
                                n_fft = 2 ** int(np.log2(min(2048, len(seg))))
                                hop   = n_fft // 4
                                mfcc  = librosa.feature.mfcc(
                                    y=seg, sr=SR, n_mfcc=n_mfcc,
                                    n_fft=n_fft, hop_length=hop).T
                                mfcc  = _normalize(mfcc)
                                if len(mfcc) < 3: continue
                                (train_data if split=='TRAIN' else test_data).append((mfcc, label))
                                count += 1
                    except Exception:
                        skipped += 1
        print(f"    {count} segments loaded from {split}")

    if skipped:
        print(f"  Skipped {skipped} files")
    return train_data, test_data


def make_balanced_subset(data, total_size):
    # If total_size is None, return the full dataset unchanged
    if total_size is None:
        return data
    class_data = defaultdict(list)
    for item in data:
        class_data[item[1]].append(item)
    per_class = max(1, total_size // len(class_data))
    subset = []
    for items in class_data.values():
        subset.extend(items[:per_class])
    return subset

# ─────────────────────────────────────────────────────────────────────────────
# Scalability benchmark — single phoneme class, vary worker count
# ─────────────────────────────────────────────────────────────────────────────

def run_scalability_benchmark(train_data):
    print(f"\n{'='*65}")
    print("SCALABILITY BENCHMARK  (single phoneme class)")
    print(f"{'='*65}")

    # Pick most frequent phoneme class for timing
    label_counts  = Counter(l for _, l in train_data)
    target_label  = label_counts.most_common(1)[0][0]
    utterances    = [f for f, l in train_data if l == target_label]

    print(f"  Phoneme: '{target_label}'  ({len(utterances)} utterances)")
    print(f"  Workers to test: {WORKER_COUNTS}\n")

    results = {}

    for n_workers in WORKER_COUNTS:
        print(f"  ── {n_workers} worker(s) " + "─"*40)
        trainer = DistributedHMMTrainer(
            n_states=N_STATES, n_features=N_MFCC,
            n_workers=n_workers, n_iter=N_ITER, seed=42
        )
        start   = time.time()
        report  = trainer.fit(utterances, verbose=True)
        elapsed = time.time() - start

        results[n_workers] = {
            'total_time':      elapsed,
            'log_likelihoods': report['log_likelihoods'],
            'iter_logs':       report['iter_logs'],
            'comm_times':      report['comm_times'],
            'final_ll':        report['log_likelihoods'][-1],
        }
        print(f"  → {n_workers} workers: {elapsed:.2f}s  "
              f"final LL={report['log_likelihoods'][-1]:.2f}\n")

    return results, target_label


# ─────────────────────────────────────────────────────────────────────────────
# Analysis tables
# ─────────────────────────────────────────────────────────────────────────────

def print_speedup_table(results):
    print(f"\n{'='*65}")
    print("SPEEDUP & EFFICIENCY TABLE")
    print(f"{'='*65}")
    print(f"  {'Workers':>7}  {'Time(s)':>9}  {'Speedup':>9}  "
          f"{'Efficiency':>11}  {'Comm(ms)':>10}")
    print("  " + "-"*55)

    baseline = results[min(results)]['total_time']

    for n in sorted(results):
        t    = results[n]['total_time']
        sp   = baseline / t
        eff  = sp / n * 100
        comm = np.mean(results[n]['comm_times']) * 1000
        print(f"  {n:>7}  {t:>9.2f}  {sp:>9.2f}x  {eff:>10.1f}%  {comm:>9.2f}ms")

    print("  " + "-"*55)


def print_communication_analysis(results):
    print(f"\n{'='*65}")
    print("COMMUNICATION OVERHEAD ANALYSIS")
    print(f"{'='*65}")

    for n in sorted(results):
        logs    = results[n]['iter_logs']
        if not logs: continue
        e_times = [l['e_step_time'] for l in logs]
        r_times = [l['reduce_time'] for l in logs]
        m_times = [l['mstep_time']  for l in logs]
        t_times = [l['iter_time']   for l in logs]

        comm_pct = np.mean(r_times) / np.mean(t_times) * 100

        print(f"\n  {n} worker(s):")
        print(f"    Avg E-step (parallel compute): {np.mean(e_times):.3f}s")
        print(f"    Avg reduction (communication): {np.mean(r_times)*1000:.2f}ms  "
              f"({comm_pct:.2f}% of total)")
        print(f"    Avg M-step:                    {np.mean(m_times)*1000:.2f}ms")
        print(f"    Avg iteration time:            {np.mean(t_times):.3f}s")

        if n > 1:
            w_times = np.array([l['worker_times'] for l in logs])
            imbal   = np.mean(np.std(w_times, axis=1) /
                              (np.mean(w_times, axis=1) + 1e-8) * 100)
            print(f"    Load imbalance (std/mean):     {imbal:.2f}%")


def verify_convergence(results):
    print(f"\n{'='*65}")
    print("CONVERGENCE VERIFICATION")
    print(f"{'='*65}")

    baseline_ll = results[min(results)]['final_ll']
    print(f"  Baseline (1 worker) final LL: {baseline_ll:.4f}\n")

    all_pass = True
    for n in sorted(results):
        ll      = results[n]['final_ll']
        diff    = abs(ll - baseline_ll)
        rel     = diff / (abs(baseline_ll) + 1e-8) * 100
        status  = "PASS" if rel < 1.0 else "WARN"
        print(f"  {n} workers: LL={ll:.4f}  |diff|={diff:.4f}  "
              f"({rel:.3f}%)  [{status}]")
        if rel >= 1.0:
            all_pass = False

    print()
    if all_pass:
        print("  RESULT: All worker counts converge correctly.")
        print("  Distributed training is mathematically equivalent to sequential.")
    else:
        print("  RESULT: Some divergence detected — investigate above entries.")


# ─────────────────────────────────────────────────────────────────────────────
# Full distributed training run (all classes)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_distributed(train_data, test_data, n_workers):
    print(f"\n{'='*65}")
    print(f"FULL DISTRIBUTED TRAINING  ({n_workers} workers, all classes)")
    print(f"{'='*65}")

    recognizer = DistributedSpeechRecognizer(
        n_states=N_STATES, n_mfcc=N_MFCC,
        n_workers=n_workers, n_iter=N_ITER
    )
    start = time.time()
    recognizer.train(train_data, verbose=False)
    train_time = time.time() - start

    print(f"\n  Training complete: {train_time:.2f}s")
    print(f"  Evaluating on {len(test_data)} test segments...")
    results = recognizer.evaluate(test_data)

    return train_time, results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  MILESTONE 2: Distributed HMM Training via Data Parallelism")
    print("  Framework: Ray   |   Workers: up to 8 logical processors")
    print("="*65)

    # ── Initialise Ray ────────────────────────────────────────────────────
    print("\n[0] Initialising Ray...")
    ray.init(num_cpus=8, ignore_reinit_error=True)
    print(f"  Ray initialised — {ray.available_resources().get('CPU', 0):.0f} CPUs available")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1] Loading TIMIT data...")
    train_data, test_data = load_timit(TIMIT_ROOT, n_mfcc=N_MFCC)

    if not train_data:
        print("ERROR: No data loaded. Check TIMIT_ROOT.")
        ray.shutdown()
        return

    # Balanced subset for benchmarking
    train_sub = make_balanced_subset(train_data, SUBSET_SIZE)
    test_sub = test_data if SUBSET_SIZE is None else test_data[:1000]
    print(f"\n  Full dataset:  {len(train_data)} train / {len(test_data)} test")
    print(f"  Benchmark subset: {len(train_sub)} train / {len(test_sub)} test")

    # ── Scalability benchmark ─────────────────────────────────────────────
    print("\n[2] Running scalability benchmark...")
    bench_results, phoneme = run_scalability_benchmark(train_sub)

    # ── Analysis ──────────────────────────────────────────────────────────
    print_speedup_table(bench_results)
    print_communication_analysis(bench_results)
    verify_convergence(bench_results)

    # ── Full distributed training ─────────────────────────────────────────
    best_workers = max(WORKER_COUNTS)
    print(f"\n[3] Full distributed training with {best_workers} workers...")
    dist_time, dist_acc = run_full_distributed(train_sub, test_sub, best_workers)

    # ── Sequential baseline comparison (1 worker = sequential) ───────────
    seq_time = bench_results[1]['total_time']
    best_t   = bench_results[best_workers]['total_time']
    speedup  = seq_time / best_t

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("MILESTONE 2 SUMMARY")
    print("="*65)
    print(f"  Framework:              Ray (shared memory transport)")
    print(f"  Logical processors:     8")
    print(f"  Max workers tested:     {best_workers}")
    print(f"  Training segments:      {len(train_sub)}")
    print()
    print(f"  Scalability (phoneme '{phoneme}'):")
    baseline = bench_results[1]['total_time']
    for n in sorted(bench_results):
        t = bench_results[n]['total_time']
        s = baseline / t
        e = s / n * 100
        print(f"    {n} worker(s): {t:.2f}s  speedup={s:.2f}x  efficiency={e:.1f}%")
    print()
    print(f"  Peak speedup:           {speedup:.2f}x  "
          f"({best_workers} workers vs 1 worker)")
    print(f"  Distributed accuracy:   {dist_acc['accuracy']*100:.1f}%")
    print(f"  Full training time:     {dist_time:.2f}s")
    print()
    print("  Milestone 2 complete. Distributed training demonstrated.")
    print("="*65)

    ray.shutdown()


if __name__ == "__main__":
    main()