"""
inference_service.py - Milestone 3: Concurrent HMM Speech Inference

Implements a concurrent inference service that decodes multiple audio
streams simultaneously using Ray remote functions.

Design:
  - Each Viterbi decode is a stateless, independent operation — perfectly
    suited for concurrent execution.
  - A worker pool of Ray actors holds model replicas to avoid re-sending
    model parameters on every request.
  - Load balancing handles variable-length audio by prioritising shortest
    segments first (Shortest Job First) to minimise mean latency.
  - Two modes are supported:
      * Batch mode:   decode a batch of segments concurrently
      * Streaming mode: continuous queue of incoming requests

Key metrics:
  - Throughput:  segments decoded per second
  - Latency:     time from request submission to result available
  - Speedup:     concurrent throughput / sequential throughput
"""

import numpy as np
import time
import ray
from typing import List, Dict, Tuple
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Numerical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_sum_exp(a, axis=None, keepdims=False):
    a_max  = np.max(a, axis=axis, keepdims=True)
    result = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


def _log_emission(features, means, covs, n_states):
    T     = len(features)
    log_B = np.zeros((T, n_states))
    for i in range(n_states):
        diff        = features - means[i]
        log_B[:, i] = -0.5 * np.sum(
            diff**2 / (covs[i] + 1e-8) + np.log(2 * np.pi * (covs[i] + 1e-8)),
            axis=1
        )
    return log_B


def _viterbi(features, log_pi, log_A, means, covs, n_states):
    """
    Viterbi decoding. Returns (state_sequence, log_prob).
    """
    T     = len(features)
    log_B = _log_emission(features, means, covs, n_states)

    delta = np.full((T, n_states), -np.inf)
    psi   = np.zeros((T, n_states), dtype=int)

    delta[0] = log_pi + log_B[0]

    for t in range(1, T):
        for j in range(n_states):
            scores    = delta[t-1] + log_A[:, j]
            best      = np.argmax(scores)
            delta[t, j] = scores[best] + log_B[t, j]
            psi[t, j]   = best

    log_prob   = np.max(delta[-1])
    last_state = np.argmax(delta[-1])

    path = [0] * T
    path[-1] = last_state
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    return path, log_prob


# ─────────────────────────────────────────────────────────────────────────────
# Ray Remote Worker — decodes one segment against all models
# ─────────────────────────────────────────────────────────────────────────────

@ray.remote
def ray_decode_segment(request_id: int, features: np.ndarray,
                       model_params: dict) -> dict:
    """
    Ray remote function: decode one segment by scoring against all HMMs.

    This is the core concurrent unit. Each call is completely independent —
    no shared state between calls, making it trivially parallelisable.

    Args:
        request_id:   integer identifier for tracking
        features:     MFCC array, shape (T, n_features)
        model_params: dict mapping label -> {log_pi, log_A, means, covs}

    Returns:
        result dict with predicted_label, score, state_path, timing info
    """
    start = time.time()

    best_label = None
    best_score = -np.inf
    best_path  = None

    for label, params in model_params.items():
        log_pi   = params['log_pi']
        log_A    = params['log_A']
        means    = params['means']
        covs     = params['covs']
        n_states = means.shape[0]

        try:
            path, score = _viterbi(features, log_pi, log_A, means, covs, n_states)
            if score > best_score:
                best_score = score
                best_label = label
                best_path  = path
        except Exception:
            continue

    elapsed = time.time() - start

    return {
        'request_id':    request_id,
        'predicted':     best_label,
        'score':         best_score,
        'state_path':    best_path,
        'n_frames':      len(features),
        'decode_time_ms': elapsed * 1000,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ray Actor — persistent worker that holds model params in memory
# ─────────────────────────────────────────────────────────────────────────────

@ray.remote
class InferenceWorker:
    """
    A Ray actor that keeps model parameters in memory.

    Unlike remote functions (which receive params on every call),
    actors initialise once and reuse params for all subsequent requests.
    This avoids the overhead of serialising model params on every decode.

    Suitable for the streaming inference mode where many requests arrive
    continuously.
    """

    def __init__(self, model_params: dict, worker_id: int):
        self.model_params = model_params
        self.worker_id    = worker_id
        self.n_decoded    = 0

    def decode(self, request_id: int, features: np.ndarray) -> dict:
        """Decode one segment using cached model params."""
        start = time.time()

        best_label = None
        best_score = -np.inf

        for label, params in self.model_params.items():
            n_states = params['means'].shape[0]
            try:
                _, score = _viterbi(
                    features, params['log_pi'], params['log_A'],
                    params['means'], params['covs'], n_states
                )
                if score > best_score:
                    best_score = score
                    best_label = label
            except Exception:
                continue

        self.n_decoded += 1
        return {
            'request_id':     request_id,
            'predicted':      best_label,
            'score':          best_score,
            'n_frames':       len(features),
            'decode_time_ms': (time.time() - start) * 1000,
            'worker_id':      self.worker_id,
        }

    def get_stats(self) -> dict:
        return {'worker_id': self.worker_id, 'n_decoded': self.n_decoded}


# ─────────────────────────────────────────────────────────────────────────────
# Load Balancing Strategies
# ─────────────────────────────────────────────────────────────────────────────

def sort_by_length_ascending(requests: list) -> list:
    """
    Shortest Job First (SJF) load balancing.

    Sort requests by number of frames ascending so shorter segments
    are dispatched first. This minimises mean latency because short
    jobs don't wait behind long ones.

    Args:
        requests: list of (request_id, features) tuples

    Returns:
        sorted list, shortest first
    """
    return sorted(requests, key=lambda x: len(x[1]))


def sort_by_length_descending(requests: list) -> list:
    """
    Longest Job First — maximises throughput by keeping workers busy
    with long jobs while short jobs fill in gaps.
    """
    return sorted(requests, key=lambda x: len(x[1]), reverse=True)


def round_robin_assign(requests: list, n_workers: int) -> list:
    """
    Assign requests to workers in round-robin order.
    Returns list of (worker_id, request_id, features) tuples.
    """
    assignments = []
    for i, (req_id, features) in enumerate(requests):
        assignments.append((i % n_workers, req_id, features))
    return assignments


def length_balanced_assign(requests: list, n_workers: int) -> list:
    """
    Assign requests to workers to balance total frame count per worker.
    Workers with fewer total frames assigned get the next request.
    This is better than round-robin when request lengths vary significantly.

    Returns list of (worker_id, request_id, features) tuples.
    """
    worker_loads = [0] * n_workers  # total frames per worker
    assignments  = []

    for req_id, features in requests:
        # Assign to least-loaded worker
        lightest   = int(np.argmin(worker_loads))
        assignments.append((lightest, req_id, features))
        worker_loads[lightest] += len(features)

    return assignments


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent Inference Service
# ─────────────────────────────────────────────────────────────────────────────

class ConcurrentInferenceService:
    """
    Concurrent speech decoding service using Ray.

    Supports two modes:
      - Batch mode: submit all requests at once, decode concurrently
      - Actor pool mode: persistent workers for streaming inference

    Load balancing strategies:
      - none:     submit in original order
      - sjf:      shortest job first (minimises mean latency)
      - ljf:      longest job first (maximises throughput)
      - balanced: assign by total frame count per worker
    """

    def __init__(self, model_params: dict, n_workers: int = 4):
        """
        Args:
            model_params: dict mapping phoneme label -> param dict
                          (log_pi, log_A, means, covs)
            n_workers:    number of concurrent Ray workers
        """
        self.model_params = model_params
        self.n_workers    = n_workers
        self._actors      = None

    def _ensure_actors(self):
        """Lazily initialise persistent actor pool."""
        if self._actors is None:
            self._actors = [
                InferenceWorker.remote(self.model_params, wid)
                for wid in range(self.n_workers)
            ]

    def decode_batch(self, requests: list,
                     load_balance: str = "sjf") -> tuple:
        """
        Decode a batch of segments concurrently using Ray remote functions.

        Each segment is dispatched as a separate Ray task. All tasks run
        in parallel up to the available CPU count.

        Args:
            requests:     list of (request_id, features) tuples
            load_balance: "none", "sjf" (shortest first), "ljf" (longest first)

        Returns:
            (results, timing_info)
            results: list of result dicts in original request order
            timing_info: dict with throughput, latency stats
        """
        if not requests:
            return [], {}

        # Apply load balancing sort
        if load_balance == "sjf":
            sorted_requests = sort_by_length_ascending(requests)
        elif load_balance == "ljf":
            sorted_requests = sort_by_length_descending(requests)
        else:
            sorted_requests = requests

        # Dispatch all tasks concurrently
        dispatch_start = time.time()
        futures = [
            ray_decode_segment.remote(req_id, features, self.model_params)
            for req_id, features in sorted_requests
        ]

        # Collect results — ray.get blocks until all complete
        raw_results    = ray.get(futures)
        total_elapsed  = time.time() - dispatch_start

        # Re-order results to match original request order
        result_map = {r['request_id']: r for r in raw_results}
        results    = [result_map[req_id] for req_id, _ in requests
                      if req_id in result_map]

        # Compute timing stats
        decode_times  = [r['decode_time_ms'] for r in raw_results]
        n             = len(requests)
        throughput    = n / total_elapsed

        timing = {
            'total_elapsed_s':   total_elapsed,
            'throughput_per_sec': throughput,
            'mean_latency_ms':   np.mean(decode_times),
            'median_latency_ms': np.median(decode_times),
            'p95_latency_ms':    np.percentile(decode_times, 95),
            'min_latency_ms':    np.min(decode_times),
            'max_latency_ms':    np.max(decode_times),
            'n_requests':        n,
            'load_balance':      load_balance,
        }

        return results, timing

    def decode_stream(self, requests: list,
                      load_balance: str = "balanced") -> tuple:
        """
        Decode using persistent actor pool — better for streaming workloads
        where requests arrive continuously and model reload overhead matters.

        Args:
            requests:     list of (request_id, features) tuples
            load_balance: "round_robin" or "balanced" (by frame count)

        Returns:
            (results, timing_info)
        """
        self._ensure_actors()

        if load_balance == "balanced":
            assignments = length_balanced_assign(requests, self.n_workers)
        else:
            assignments = round_robin_assign(requests, self.n_workers)

        # Dispatch to actors
        dispatch_start = time.time()
        futures = []
        for worker_id, req_id, features in assignments:
            fut = self._actors[worker_id].decode.remote(req_id, features)
            futures.append(fut)

        raw_results   = ray.get(futures)
        total_elapsed = time.time() - dispatch_start

        result_map = {r['request_id']: r for r in raw_results}
        results    = [result_map[req_id] for req_id, _ in requests
                      if req_id in result_map]

        decode_times = [r['decode_time_ms'] for r in raw_results]
        n            = len(requests)

        timing = {
            'total_elapsed_s':    total_elapsed,
            'throughput_per_sec': n / total_elapsed,
            'mean_latency_ms':    np.mean(decode_times),
            'median_latency_ms':  np.median(decode_times),
            'p95_latency_ms':     np.percentile(decode_times, 95),
            'min_latency_ms':     np.min(decode_times),
            'max_latency_ms':     np.max(decode_times),
            'n_requests':         n,
            'load_balance':       load_balance,
        }

        return results, timing

    def shutdown(self):
        """Kill actor pool."""
        if self._actors:
            for actor in self._actors:
                ray.kill(actor)
            self._actors = None


def extract_model_params(recognizer) -> dict:
    """
    Extract model parameters from a trained DistributedSpeechRecognizer
    or sequential SpeechRecognizer into a plain dict for the inference service.
    """
    params = {}
    for label, model in recognizer.models.items():
        if hasattr(model, 'get_params'):
            p = model.get_params()
        else:
            # Sequential GaussianHMM
            p = {
                'log_pi': model.log_pi.copy(),
                'log_A':  model.log_A.copy(),
                'means':  model.means.copy(),
                'covs':   model.covs.copy(),
            }
        params[label] = p
    return params