"""
inference_main.py - Milestone 3: Concurrent Speech Inference Entry Point

Runs the complete inference benchmark:
  1. Load TIMIT test data
  2. Load trained models (from Milestone 2 distributed training)
  3. Sequential baseline — decode one segment at a time
  4. Concurrent batch — decode all segments simultaneously
  5. Actor pool streaming — persistent workers, balanced load
  6. Load balancing comparison — SJF vs LJF vs no sorting
  7. Scalability — vary worker count 1→2→4→8
  8. Varying workload — different batch sizes

Run:
    python inference_main.py
"""

import os
import sys
import time
import numpy as np
import ray
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distributed_trainer import DistributedSpeechRecognizer
from inference_service import (
    ConcurrentInferenceService,
    extract_model_params,
    _viterbi,
    _log_emission
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TIMIT_ROOT    = r"C:\Users\fahad\OneDrive\Documents\IBA\spring26\PDC\HMM_Project\timit"
N_MFCC        = 13
N_STATES      = 5
N_WORKERS     = 8       # max workers for inference

# Batch sizes to test for workload variation benchmark
BATCH_SIZES   = [50, 100, 250, 500, 1000]

# Worker counts for scalability test
WORKER_COUNTS = [1, 2, 4, 8]

# Training subset size (keep small for faster training in this demo)
TRAIN_SUBSET  = None
TEST_SUBSET   = None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same as Milestone 2)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(features):
    mean = features.mean(axis=0)
    std  = features.std(axis=0) + 1e-8
    return (features - mean) / std


def load_timit(timit_root, n_mfcc=13):
    import librosa
    train_data, test_data = [], []
    SR = 16000
    for split in ['TRAIN', 'TEST']:
        split_dir = os.path.join(timit_root, split)
        if not os.path.exists(split_dir):
            continue
        print(f"  Loading {split}...")
        count = 0
        for dr in sorted(os.listdir(split_dir)):
            dr_dir = os.path.join(split_dir, dr)
            if not os.path.isdir(dr_dir): continue
            for speaker in sorted(os.listdir(dr_dir)):
                spk_dir = os.path.join(dr_dir, speaker)
                if not os.path.isdir(spk_dir): continue
                for fname in sorted(os.listdir(spk_dir)):
                    if not fname.upper().endswith('.WAV'): continue
                    wav  = os.path.join(spk_dir, fname)
                    phn  = os.path.join(spk_dir, fname.upper().replace('.WAV','.PHN'))
                    if not os.path.exists(phn):
                        phn = os.path.join(spk_dir, fname.lower().replace('.wav','.phn'))
                    if not os.path.exists(phn): continue
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
                        pass
        print(f"    {count} segments loaded")
    return train_data, test_data


def make_balanced_subset(data, total_size):
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
# Sequential baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_sequential_baseline(model_params: dict, test_data: list) -> dict:
    """
    Decode test segments one at a time sequentially.
    This is the baseline for speedup comparison.
    """
    print(f"\n  Sequential baseline ({len(test_data)} segments)...")
    times   = []
    correct = 0

    for features, true_label in test_data:
        start = time.time()

        best_label = None
        best_score = -np.inf

        for label, params in model_params.items():
            n_states = params['means'].shape[0]
            try:
                log_B = _log_emission(features, params['means'],
                                      params['covs'], n_states)
                T = len(features)
                log_alpha = np.zeros((T, n_states))
                log_alpha[0] = params['log_pi'] + log_B[0]
                for t in range(1, T):
                    for j in range(n_states):
                        from inference_service import _log_sum_exp
                        log_alpha[t, j] = (_log_sum_exp(
                            log_alpha[t-1] + params['log_A'][:, j]) + log_B[t, j])
                score = float(np.max(log_alpha[-1]) +
                              np.log(np.sum(np.exp(log_alpha[-1] - np.max(log_alpha[-1])))))
                if score > best_score:
                    best_score = score
                    best_label = label
            except Exception:
                continue

        elapsed = time.time() - start
        times.append(elapsed * 1000)

        if str(best_label) == str(true_label):
            correct += 1

    times = np.array(times)
    acc   = correct / len(test_data)

    print(f"    Accuracy:    {acc*100:.1f}%")
    print(f"    Throughput:  {1000/times.mean():.1f} segments/sec")
    print(f"    Mean latency: {times.mean():.2f} ms")

    return {
        'accuracy':          acc,
        'throughput_per_sec': 1000 / times.mean(),
        'mean_latency_ms':   times.mean(),
        'median_latency_ms': np.median(times),
        'p95_latency_ms':    np.percentile(times, 95),
        'total_time_s':      times.sum() / 1000,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_concurrent_benchmark(service: ConcurrentInferenceService,
                              test_data: list,
                              load_balance: str = "sjf",
                              label: str = "") -> dict:
    requests = [(i, feat) for i, (feat, _) in enumerate(test_data)]
    true_labels = {i: lbl for i, (_, lbl) in enumerate(test_data)}

    results, timing = service.decode_batch(requests, load_balance=load_balance)

    correct = sum(1 for r in results
                  if str(r['predicted']) == str(true_labels.get(r['request_id'], '')))
    acc = correct / len(results) if results else 0

    tag = label or f"concurrent ({load_balance})"
    print(f"    [{tag}]  "
          f"throughput={timing['throughput_per_sec']:.1f}/s  "
          f"mean={timing['mean_latency_ms']:.2f}ms  "
          f"p95={timing['p95_latency_ms']:.2f}ms  "
          f"acc={acc*100:.1f}%")

    timing['accuracy'] = acc
    return timing


def run_load_balance_comparison(model_params, test_data):
    """Compare SJF, LJF, and no-sorting load balancing."""
    print(f"\n{'='*65}")
    print("LOAD BALANCING STRATEGY COMPARISON")
    print(f"{'='*65}")
    print(f"  Segments: {len(test_data)}  Workers: {N_WORKERS}\n")

    results = {}
    for strategy in ["none", "sjf", "ljf"]:
        service = ConcurrentInferenceService(model_params, n_workers=N_WORKERS)
        requests = [(i, feat) for i, (feat, _) in enumerate(test_data)]
        _, timing = service.decode_batch(requests, load_balance=strategy)
        results[strategy] = timing
        print(f"  {strategy.upper():6s}: throughput={timing['throughput_per_sec']:.1f}/s  "
              f"mean={timing['mean_latency_ms']:.2f}ms  "
              f"p95={timing['p95_latency_ms']:.2f}ms")

    print()
    best    = max(results, key=lambda s: results[s]['throughput_per_sec'])
    print(f"  Best strategy: {best.upper()} "
          f"({results[best]['throughput_per_sec']:.1f} seg/s)")
    return results


def run_worker_scalability(model_params, test_data):
    """Vary worker count and measure throughput."""
    print(f"\n{'='*65}")
    print("WORKER SCALABILITY  (concurrent inference)")
    print(f"{'='*65}")
    print(f"  Segments: {len(test_data)}\n")

    results  = {}
    requests = [(i, feat) for i, (feat, _) in enumerate(test_data)]

    for n_workers in WORKER_COUNTS:
        service = ConcurrentInferenceService(model_params, n_workers=n_workers)
        _, timing = service.decode_batch(requests, load_balance="sjf")
        results[n_workers] = timing
        print(f"  {n_workers} workers: {timing['throughput_per_sec']:.1f} seg/s  "
              f"mean={timing['mean_latency_ms']:.2f}ms  "
              f"p95={timing['p95_latency_ms']:.2f}ms")

    return results


def run_batch_size_benchmark(model_params, test_data):
    """Measure throughput and latency under varying workload sizes."""
    print(f"\n{'='*65}")
    print("VARYING WORKLOAD BENCHMARK")
    print(f"{'='*65}")
    print(f"  Workers: {N_WORKERS}\n")

    results  = {}
    service  = ConcurrentInferenceService(model_params, n_workers=N_WORKERS)

    for batch_size in BATCH_SIZES:
        batch    = test_data[:batch_size]
        requests = [(i, feat) for i, (feat, _) in enumerate(batch)]
        _, timing = service.decode_batch(requests, load_balance="sjf")
        results[batch_size] = timing
        print(f"  {batch_size:5d} segments: "
              f"throughput={timing['throughput_per_sec']:.1f}/s  "
              f"mean={timing['mean_latency_ms']:.2f}ms  "
              f"total={timing['total_elapsed_s']:.2f}s")

    return results


def print_speedup_table(seq_baseline, worker_results):
    """Print speedup table comparing sequential vs concurrent."""
    print(f"\n{'='*65}")
    print("SPEEDUP TABLE  (concurrent vs sequential)")
    print(f"{'='*65}")
    print(f"  {'Config':>16}  {'Throughput':>12}  {'Speedup':>10}  "
          f"{'Mean Lat':>10}  {'p95 Lat':>10}")
    print("  " + "-"*60)

    seq_tp = seq_baseline['throughput_per_sec']
    print(f"  {'Sequential':>16}  {seq_tp:>11.1f}/s  {'1.00x':>10}  "
          f"{seq_baseline['mean_latency_ms']:>9.2f}ms  "
          f"{seq_baseline['p95_latency_ms']:>9.2f}ms")

    for n_workers, timing in sorted(worker_results.items()):
        tp      = timing['throughput_per_sec']
        speedup = tp / seq_tp
        print(f"  {f'{n_workers} workers':>16}  {tp:>11.1f}/s  "
              f"{speedup:>9.2f}x  "
              f"{timing['mean_latency_ms']:>9.2f}ms  "
              f"{timing['p95_latency_ms']:>9.2f}ms")

    print("  " + "-"*60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  MILESTONE 3: Concurrent HMM Speech Inference")
    print("  Framework: Ray   |   Workers: up to 8")
    print("="*65)

    # ── Init Ray ──────────────────────────────────────────────────────────
    print("\n[0] Initialising Ray...")
    ray.init(num_cpus=8, ignore_reinit_error=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1] Loading TIMIT data...")
    train_data, test_data = load_timit(TIMIT_ROOT, n_mfcc=N_MFCC)

    if not train_data:
        print("ERROR: No data loaded.")
        ray.shutdown()
        return

    train_sub = make_balanced_subset(train_data, TRAIN_SUBSET)
    test_sub  = test_data if TEST_SUBSET is None else test_data[:TEST_SUBSET]
    print(f"\n  Training subset: {len(train_sub)} segments")
    print(f"  Test subset:     {len(test_sub)} segments")

    # ── Train models (distributed, Milestone 2 style) ─────────────────────
    print("\n[2] Training HMMs (distributed, 8 workers)...")
    recognizer = DistributedSpeechRecognizer(
        n_states=N_STATES, n_mfcc=N_MFCC,
        n_workers=8, n_iter=10
    )
    recognizer.train(train_sub, verbose=False)

    # Extract params for inference service
    model_params = extract_model_params(recognizer)
    print(f"  Models trained: {len(model_params)} phoneme classes")

    # ── Sequential baseline ────────────────────────────────────────────────
    print("\n[3] Sequential inference baseline...")
    seq_baseline = run_sequential_baseline(model_params, test_sub)

    # ── Concurrent batch inference ─────────────────────────────────────────
    print(f"\n[4] Concurrent batch inference ({N_WORKERS} workers)...")
    service = ConcurrentInferenceService(model_params, n_workers=N_WORKERS)
    requests = [(i, feat) for i, (feat, _) in enumerate(test_sub)]
    batch_results, batch_timing = service.decode_batch(requests, load_balance="sjf")
    true_labels = {i: lbl for i, (_, lbl) in enumerate(test_sub)}
    correct = sum(1 for r in batch_results
                  if str(r['predicted']) == str(true_labels.get(r['request_id'], '')))
    batch_acc = correct / len(batch_results)
    print(f"  Throughput:  {batch_timing['throughput_per_sec']:.1f} seg/s")
    print(f"  Mean latency: {batch_timing['mean_latency_ms']:.2f} ms")
    print(f"  Accuracy:    {batch_acc*100:.1f}%")

    # ── Actor pool streaming inference ────────────────────────────────────
    print(f"\n[5] Actor pool streaming inference ({N_WORKERS} workers)...")
    stream_results, stream_timing = service.decode_stream(requests, load_balance="balanced")
    print(f"  Throughput:  {stream_timing['throughput_per_sec']:.1f} seg/s")
    print(f"  Mean latency: {stream_timing['mean_latency_ms']:.2f} ms")

    # ── Load balancing comparison ──────────────────────────────────────────
    lb_results = run_load_balance_comparison(model_params, test_sub)

    # ── Worker scalability ────────────────────────────────────────────────
    worker_results = run_worker_scalability(model_params, test_sub)

    # ── Batch size benchmark ──────────────────────────────────────────────
    batch_size_results = run_batch_size_benchmark(model_params, test_sub)

    # ── Speedup table ─────────────────────────────────────────────────────
    print_speedup_table(seq_baseline, worker_results)

    # ── Final summary ──────────────────────────────────────────────────────
    best_tp      = max(worker_results[n]['throughput_per_sec'] for n in worker_results)
    best_workers = max(worker_results, key=lambda n: worker_results[n]['throughput_per_sec'])
    speedup      = best_tp / seq_baseline['throughput_per_sec']

    print("\n" + "="*65)
    print("MILESTONE 3 SUMMARY")
    print("="*65)
    print(f"  Framework:              Ray")
    print(f"  Max workers:            {N_WORKERS}")
    print(f"  Test segments:          {len(test_sub)}")
    print()
    print(f"  Sequential baseline:")
    print(f"    Throughput:           {seq_baseline['throughput_per_sec']:.1f} seg/s")
    print(f"    Mean latency:         {seq_baseline['mean_latency_ms']:.2f} ms")
    print()
    print(f"  Best concurrent ({best_workers} workers):")
    print(f"    Throughput:           {best_tp:.1f} seg/s")
    print(f"    Mean latency:         {worker_results[best_workers]['mean_latency_ms']:.2f} ms")
    print(f"    Speedup:              {speedup:.2f}x")
    print()
    print(f"  Accuracy (concurrent):  {batch_acc*100:.1f}%")
    print()
    print("  Milestone 3 complete. Concurrent inference demonstrated.")
    print("="*65)

    service.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    main()