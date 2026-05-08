# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — HMM Speech Recognition (PDC Group 9)
# Provides a controlled environment for reproducible benchmarking.
#
# Build:
#   docker build -t hmm-speech-recognition .
#
# Run full benchmark (all 3 milestones):
#   docker run --rm --cpus="4" \
#     -v /path/to/timit:/app/timit \
#     hmm-speech-recognition \
#     python run_all_benchmarks.py
#
# Run individual milestones:
#   docker run --rm --cpus="4" -v /path/to/timit:/app/timit hmm-speech-recognition python main.py
#   docker run --rm --cpus="4" -v /path/to/timit:/app/timit hmm-speech-recognition python distributed_main.py
#   docker run --rm --cpus="4" -v /path/to/timit:/app/timit hmm-speech-recognition python inference_main.py
#
# Windows path example:
#   docker run --rm --cpus="4" \
#     -v "C:/Users/fahad/.../timit:/app/timit" \
#     hmm-speech-recognition \
#     python run_all_benchmarks.py
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by librosa and soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    librosa \
    soundfile \
    sphfile \
    ray \
    matplotlib

# Copy all project Python files
COPY *.py .

# Default: run the unified benchmark runner
CMD ["python", "run_all_benchmarks.py"]