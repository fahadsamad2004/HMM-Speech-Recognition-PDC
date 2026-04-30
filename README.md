# Distributed HMM Speech Recognition
Parallel & Distributed Computing — Group 9

Fahad Samad | Fahad Shah Khan | Muhammad Hussain | Saman Waseem

## Milestone 1 — Sequential HMM Pipeline
python main.py

## Milestone 2 — Distributed Training via Ray
python distributed_main.py

## Milestone 3 — Concurrent Inference Service
python inference_main.py

## Setup
conda create -n ray_env python=3.12
conda activate ray_env
pip install -r requirements.txt

## Note
TIMIT dataset not included due to licensing restrictions.
Update TIMIT_ROOT path in main.py, distributed_main.py,
and inference_main.py to your local TIMIT folder path.