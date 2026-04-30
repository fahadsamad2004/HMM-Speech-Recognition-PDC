"""
main.py - Milestone 1: Sequential HMM Speech Recognition
Uses real TIMIT data for training and evaluation.
"""


import os
import numpy as np
import time
from features import extract_mfcc, normalize_features
from pipeline import SpeechRecognizer


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — update this path to your TIMIT root folder
# ─────────────────────────────────────────────────────────────────────────────
TIMIT_ROOT = r"C:\Users\fahad\OneDrive\Documents\IBA\spring26\PDC\HMM_Project\timit"
N_MFCC     = 13   # number of MFCC coefficients
N_STATES   = 5    # HMM states per phoneme class
N_ITER     = 25   # Baum-Welch iterations


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def read_phn_file(phn_path: str) -> list:
    """
    Read a TIMIT .PHN file and return a list of phoneme labels.
    Each line has the format:  start_sample  end_sample  phoneme
    """
    labels = []
    with open(phn_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                labels.append(parts[2])
    return labels


def load_timit(timit_root: str, n_mfcc: int = 13) -> tuple:
    import librosa

    train_data = []
    test_data  = []
    skipped    = 0
    SR         = 16000  # TIMIT sample rate

    for split in ['TRAIN', 'TEST']:
        split_dir = os.path.join(timit_root, split)
        if not os.path.exists(split_dir):
            print(f"  Warning: {split_dir} not found, skipping.")
            continue

        print(f"  Loading {split} (segment level)...")
        count = 0

        for dr in sorted(os.listdir(split_dir)):
            dr_dir = os.path.join(split_dir, dr)
            if not os.path.isdir(dr_dir):
                continue

            for speaker in sorted(os.listdir(dr_dir)):
                speaker_dir = os.path.join(dr_dir, speaker)
                if not os.path.isdir(speaker_dir):
                    continue

                for fname in sorted(os.listdir(speaker_dir)):
                    if not fname.upper().endswith('.WAV'):
                        continue

                    wav_path = os.path.join(speaker_dir, fname)
                    phn_path = os.path.join(speaker_dir,
                                fname.upper().replace('.WAV', '.PHN'))
                    if not os.path.exists(phn_path):
                        phn_path = os.path.join(speaker_dir,
                                    fname.lower().replace('.wav', '.phn'))
                    if not os.path.exists(phn_path):
                        skipped += 1
                        continue

                    try:
                        # Load full audio once per file
                        y, _ = librosa.load(wav_path, sr=SR)

                        # Read phoneme segments
                        with open(phn_path) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) != 3:
                                    continue

                                start_sample = int(parts[0])
                                end_sample   = int(parts[1])
                                label        = parts[2]

                                # Skip silence
                                if label == 'h#':
                                    continue

                                # Extract segment audio
                                segment = y[start_sample:end_sample]

                                # Skip segments that are too short (less than 20ms)
                                if len(segment) < SR * 0.02:
                                    continue

                                # Automatically shrink n_fft for short segments
                                n_fft = min(2048, len(segment))
                                # n_fft must be a power of 2 for efficiency
                                n_fft = 2 ** int(np.log2(n_fft))
                                hop_length = n_fft // 4

                                # Extract MFCCs from segment
                                mfcc = librosa.feature.mfcc(
                                    y=segment, sr=SR, n_mfcc=n_mfcc,
                                    n_fft=n_fft, hop_length=hop_length
                                ).T
                                mfcc = normalize_features(mfcc)

                                # Need at least 3 frames for HMM
                                if len(mfcc) < 3:
                                    continue

                                if split == 'TRAIN':
                                    train_data.append((mfcc, label))
                                else:
                                    test_data.append((mfcc, label))

                                count += 1

                    except Exception:
                        skipped += 1
                        continue

        print(f"    Loaded {count} segments from {split}")

    if skipped > 0:
        print(f"  Skipped {skipped} files")

    return train_data, test_data
    
# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_training(train_data: list, n_states: int = 5):
    """
    Measure training time at 25%, 50%, and 100% of the dataset.
    These numbers are the sequential baseline for Milestone 2 speedup analysis.
    """
    print("\n" + "=" * 60)
    print("TRAINING BENCHMARK")
    print("=" * 60)

    results = []
    for frac in [0.25, 0.5, 1.0]:
        subset_size = max(1, int(len(train_data) * frac))
        subset = train_data[:subset_size]

        recognizer = SpeechRecognizer(n_states=n_states, n_iter=10)
        start   = time.time()
        recognizer.train(subset, verbose=False)
        elapsed = time.time() - start

        results.append((subset_size, elapsed))
        print(f"  {subset_size:>5} utterances: {elapsed:.2f}s")

    return results


def benchmark_inference(recognizer: SpeechRecognizer, test_data: list) -> dict:
    """
    Measure per-utterance decode latency and overall throughput.
    These numbers are the sequential baseline for Milestone 3 speedup analysis.
    """
    print("\n" + "=" * 60)
    print("INFERENCE BENCHMARK")
    print("=" * 60)

    times = []
    for features, _ in test_data:
        start = time.time()
        recognizer.predict_features(features)
        times.append(time.time() - start)

    times = np.array(times) * 1000  # ms

    print(f"  Total utterances decoded: {len(times)}")
    print(f"  Throughput:               {1000 / times.mean():.1f} utterances/second")
    print(f"  Mean latency:             {times.mean():.2f} ms")
    print(f"  Median latency:           {np.median(times):.2f} ms")
    print(f"  Min / Max latency:        {times.min():.2f} / {times.max():.2f} ms")
    print(f"  95th percentile latency:  {np.percentile(times, 95):.2f} ms")

    return {
        'mean_latency_ms':    times.mean(),
        'throughput_per_sec': 1000 / times.mean(),
        'p95_ms':             np.percentile(times, 95),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  MILESTONE 1: Sequential HMM Speech Recognition")
    print("=" * 60)

    # ── 1. Load TIMIT data ────────────────────────────────────────────────
    print("\n[1] Loading TIMIT data...")
    train_data, test_data = load_timit(TIMIT_ROOT, n_mfcc=N_MFCC)

    if not train_data or not test_data:
        print("\nERROR: No data loaded. Check that TIMIT_ROOT path is correct.")
        print(f"  Current path: {TIMIT_ROOT}")
        return

    # Report unique phoneme classes found
    all_labels = list(set(label for _, label in train_data))
    n_classes  = len(all_labels)

    print(f"\n  Training utterances:          {len(train_data)}")
    print(f"  Test utterances:              {len(test_data)}")
    print(f"  Unique phoneme classes found: {n_classes}")
    print(f"  Classes: {sorted(all_labels)}")

    # ── 2. Train ──────────────────────────────────────────────────────────
    print("\n[2] Training HMMs (Baum-Welch)...")
    print(f"    One HMM per phoneme class  ({n_classes} models x {N_STATES} states)")

    recognizer = SpeechRecognizer(
        n_states=N_STATES,
        n_mfcc=N_MFCC,
        n_iter=N_ITER
    )

    train_start = time.time()
    recognizer.train(train_data, verbose=True)
    train_time  = time.time() - train_start

    # ── 3. Evaluate ───────────────────────────────────────────────────────
    print("\n[3] Evaluating on test set...")
    results = recognizer.evaluate(test_data, verbose=True)

    # ── 4. Sample Viterbi path ────────────────────────────────────────────
    print("\n[4] Sample Viterbi decoding (first test utterance)...")
    sample_features, sample_label = test_data[0]
    print(f"    True label: {sample_label}")
    recognizer.decode_state_path(sample_features, label=sample_label)

    # ── 5. Benchmarks ─────────────────────────────────────────────────────
    print("\n[5] Running benchmarks...")
    benchmark_training(train_data, n_states=N_STATES)
    infer_bench = benchmark_inference(recognizer, test_data)

    # ── 6. Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MILESTONE 1 SUMMARY")
    print("=" * 60)
    print(f"  Dataset:              TIMIT ({len(train_data)} train / {len(test_data)} test)")
    print(f"  Phoneme classes:      {n_classes}")
    print(f"  Accuracy:             {results['accuracy'] * 100:.1f}%")
    print(f"  Total training time:  {train_time:.2f}s")
    print(f"  Avg inference time:   {results['avg_decode_time_ms']:.1f} ms")
    print(f"  Throughput:           {infer_bench['throughput_per_sec']:.1f} utterances/sec")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()