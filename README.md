# Distributed HMM Speech Recognition
**Parallel & Distributed Computing Course — Group 9**

Fahad Samad | Fahad Shah Khan | Muhammad Hussain | Saman Waseem

---

## Project Overview
Distributed training and concurrent inference of Hidden Markov Models
for phoneme-level speech recognition using the TIMIT dataset and Ray.
All benchmarks are fully reproducible inside Docker.

---

## Milestones

| Milestone | Description | Entry Point |
|---|---|---|
| 1 | Sequential HMM pipeline (Baum-Welch + Viterbi) | `python main.py` |
| 2 | Distributed training via Ray data parallelism | `python distributed_main.py` |
| 3 | Concurrent inference service (batch + actor pool) | `python inference_main.py` |

---

## Key Results (Docker, Full TIMIT Dataset)

| Metric | Value |
|---|---|
| Dataset | 160,378 train / 58,262 test segments |
| Phoneme classes | 60 |
| M1 Sequential accuracy | 22.7% |
| M1 Training time | 1,779.19 s |
| M2 Distributed accuracy | 22.7% (identical to sequential) |
| M2 Full training time (8 workers) | 849.99 s |
| M2 Peak speedup | 2.87x (4 workers) |
| M2 End-to-end speedup vs M1 | 2.09x |
| M2 Convergence | PASS — 0.000% LL difference |
| M3 Sequential inference | 53.2 seg/s |
| M3 Actor pool inference | 238.8 seg/s (4.49x speedup) |
| Benchmark environment | Docker, Linux WSL2, Python 3.12 |

---

## Setup

```bash
conda create -n ray_env python=3.12
conda activate ray_env
pip install -r requirements.txt
```

## Dataset
Place the TIMIT dataset at your local path and update `TIMIT_ROOT`
in `main.py`, `distributed_main.py`, and `inference_main.py`.

---

## Running Locally

```bash
conda activate ray_env

# Milestone 1 — Sequential baseline
python main.py

# Milestone 2 — Distributed training
python distributed_main.py

# Milestone 3 — Concurrent inference
python inference_main.py
```

---

## Running in Docker (Reproducible Benchmarks)

```bash
# Build the image (one time)
docker build -t hmm-speech-recognition .

# Run all three milestones end-to-end
docker run --rm --cpus="4" --shm-size=2g \
  -v "C:/path/to/timit:/app/timit" \
  hmm-speech-recognition \
  python run_all_benchmarks.py

# Run individual milestones
docker run --rm --cpus="4" --shm-size=2g \
  -v "C:/path/to/timit:/app/timit" \
  hmm-speech-recognition python main.py

docker run --rm --cpus="4" --shm-size=2g \
  -v "C:/path/to/timit:/app/timit" \
  hmm-speech-recognition python distributed_main.py

docker run --rm --cpus="4" --shm-size=2g \
  -v "C:/path/to/timit:/app/timit" \
  hmm-speech-recognition python inference_main.py
```

---

## File Structure
├── features.py              # MFCC extraction and normalisation
├── hmm.py                   # GaussianHMM — Baum-Welch training
├── viterbi.py               # Viterbi decoder and classification
├── pipeline.py              # SpeechRecognizer — end-to-end pipeline
├── main.py                  # Milestone 1 entry point
├── distributed_trainer.py   # Ray distributed Baum-Welch
├── distributed_main.py      # Milestone 2 entry point
├── inference_service.py     # Concurrent inference service
├── inference_main.py        # Milestone 3 entry point
├── run_all_benchmarks.py    # Unified Docker benchmark runner
├── Dockerfile               # Docker environment definition
├── requirements.txt         # Python dependencies
└── README.md

---

## Note
TIMIT dataset not included due to licensing restrictions.
Set `TIMIT_ROOT` in each entry point file to your local TIMIT path.
Inside Docker, mount your TIMIT folder to `/app/timit`.