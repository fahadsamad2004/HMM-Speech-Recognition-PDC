"""
distributed_trainer.py - Milestone 2: Distributed HMM Training using Ray

Architecture:
  - Ray remote functions handle the E-step per worker
  - Shared memory used for large numpy arrays (no pickle overhead)
  - Main process owns model parameters and runs M-step
  - Reduction aggregates statistics from all workers after each iteration

Data parallelism flow per iteration:
  1. PARTITION  — split utterances across N workers
  2. E-STEP     — workers run forward-backward in parallel (Ray remote)
  3. REDUCE     — main process aggregates sufficient statistics
  4. M-STEP     — main process updates global HMM parameters
  5. REPEAT     — until convergence
"""

import numpy as np
import time
import ray
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Numerical helpers — defined at module level so Ray workers can access them
# ─────────────────────────────────────────────────────────────────────────────

def _log_sum_exp(a, axis=None, keepdims=False):
    """Numerically stable log-sum-exp."""
    a_max  = np.max(a, axis=axis, keepdims=True)
    result = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


def _log_emission(features, means, covs, n_states):
    """Compute log emission prob matrix shape (T, n_states)."""
    T     = len(features)
    log_B = np.zeros((T, n_states))
    for i in range(n_states):
        diff       = features - means[i]
        log_B[:, i] = -0.5 * np.sum(
            diff**2 / (covs[i] + 1e-8) + np.log(2 * np.pi * (covs[i] + 1e-8)),
            axis=1
        )
    return log_B


def _forward_backward(features, log_pi, log_A, means, covs, n_states, n_features):
    """
    Run forward-backward on one utterance.
    Returns gamma, xi, and log-likelihood.
    All computation in log-space for numerical stability.
    """
    T     = len(features)
    log_B = _log_emission(features, means, covs, n_states)

    # Forward pass
    log_alpha      = np.zeros((T, n_states))
    log_alpha[0]   = log_pi + log_B[0]
    for t in range(1, T):
        for j in range(n_states):
            log_alpha[t, j] = (_log_sum_exp(log_alpha[t-1] + log_A[:, j])
                               + log_B[t, j])
    log_ll = _log_sum_exp(log_alpha[-1])

    # Backward pass
    log_beta     = np.zeros((T, n_states))
    for t in range(T - 2, -1, -1):
        for i in range(n_states):
            log_beta[t, i] = _log_sum_exp(
                log_A[i] + log_B[t+1] + log_beta[t+1]
            )

    # Gamma — state occupation probabilities
    log_gamma  = log_alpha + log_beta
    log_gamma -= _log_sum_exp(log_gamma, axis=1, keepdims=True)
    gamma      = np.exp(log_gamma)

    # Xi — transition probabilities
    log_xi = np.zeros((T-1, n_states, n_states))
    for t in range(T - 1):
        for i in range(n_states):
            for j in range(n_states):
                log_xi[t, i, j] = (log_alpha[t, i] + log_A[i, j]
                                   + log_B[t+1, j] + log_beta[t+1, j])
        log_xi[t] -= _log_sum_exp(log_xi[t].ravel())
    xi = np.exp(log_xi)

    return gamma, xi, log_ll


# ─────────────────────────────────────────────────────────────────────────────
# Ray Remote Worker — E-step for a partition of utterances
# ─────────────────────────────────────────────────────────────────────────────

@ray.remote
def ray_worker_e_step(worker_id: int, model_params: dict,
                      utterances: list) -> dict:
    """
    Ray remote function: compute sufficient statistics for one data partition.

    This runs in a separate Ray worker process. Ray serialises the inputs
    using Apache Arrow / shared memory, which is faster than pickle for
    large numpy arrays.

    Args:
        worker_id:    integer identifier
        model_params: dict with log_pi, log_A, means, covs
        utterances:   list of feature arrays

    Returns:
        dict of accumulated sufficient statistics
    """
    start_time  = time.time()

    log_pi    = model_params['log_pi']
    log_A     = model_params['log_A']
    means     = model_params['means']
    covs      = model_params['covs']
    n_states  = means.shape[0]
    n_features= means.shape[1]

    # Accumulators
    gamma_sum      = np.zeros(n_states)
    gamma_init     = np.zeros(n_states)
    xi_sum         = np.zeros((n_states, n_states))
    gamma_obs_sum  = np.zeros((n_states, n_features))
    gamma_obs2_sum = np.zeros((n_states, n_features))
    total_ll       = 0.0

    for features in utterances:
        if len(features) < 2:
            continue

        gamma, xi, log_ll = _forward_backward(
            features, log_pi, log_A, means, covs, n_states, n_features
        )

        total_ll       += log_ll
        gamma_sum      += gamma.sum(axis=0)
        gamma_init     += gamma[0]
        xi_sum         += xi.sum(axis=0)

        for i in range(n_states):
            w = gamma[:, i:i+1]
            gamma_obs_sum[i]  += (w * features).sum(axis=0)
            gamma_obs2_sum[i] += (w * features**2).sum(axis=0)

    return {
        'gamma_sum':       gamma_sum,
        'gamma_init':      gamma_init,
        'xi_sum':          xi_sum,
        'gamma_obs_sum':   gamma_obs_sum,
        'gamma_obs2_sum':  gamma_obs2_sum,
        'total_ll':        total_ll,
        'n_utterances':    len(utterances),
        'worker_id':       worker_id,
        'elapsed':         time.time() - start_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Distributed HMM Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DistributedHMMTrainer:
    """
    Trains a Gaussian HMM using Ray-based data-parallel Baum-Welch.

    The E-step is distributed across N Ray workers. The reduction and
    M-step run on the main process. Communication cost is one round-trip
    of sufficient statistics per iteration — O(n_states^2 + n_states*n_features)
    regardless of dataset size.
    """

    def __init__(self, n_states: int, n_features: int,
                 n_workers: int = 4, n_iter: int = 20,
                 tol: float = 1e-4, seed: int = 42):
        self.n_states   = n_states
        self.n_features = n_features
        self.n_workers  = n_workers
        self.n_iter     = n_iter
        self.tol        = tol

        # Random parameter initialisation
        rng         = np.random.RandomState(seed)
        pi          = rng.dirichlet(np.ones(n_states))
        A           = rng.dirichlet(np.ones(n_states), size=n_states)
        self.log_pi = np.log(pi + 1e-300)
        self.log_A  = np.log(A  + 1e-300)
        self.means  = rng.randn(n_states, n_features)
        self.covs   = np.ones((n_states, n_features))

    # ── Data Partitioning ─────────────────────────────────────────────────

    def _partition(self, utterances: list) -> list:
        """
        Round-robin partition of utterances across workers.
        Ensures balanced load even when len(utterances) % n_workers != 0.
        """
        partitions = [[] for _ in range(self.n_workers)]
        for i, utt in enumerate(utterances):
            partitions[i % self.n_workers].append(utt)
        return partitions

    # ── Reduction step ────────────────────────────────────────────────────

    def _reduce(self, worker_results: list) -> dict:
        """
        Aggregate sufficient statistics from all workers.

        This is a simple summation — the result is mathematically identical
        to computing statistics sequentially on the full dataset.
        Communication cost: O(n_states^2 + n_states * n_features) per worker.
        """
        agg = {
            'gamma_sum':       np.zeros(self.n_states),
            'gamma_init':      np.zeros(self.n_states),
            'xi_sum':          np.zeros((self.n_states, self.n_states)),
            'gamma_obs_sum':   np.zeros((self.n_states, self.n_features)),
            'gamma_obs2_sum':  np.zeros((self.n_states, self.n_features)),
            'total_ll':        0.0,
            'n_utterances':    0,
        }
        for r in worker_results:
            agg['gamma_sum']      += r['gamma_sum']
            agg['gamma_init']     += r['gamma_init']
            agg['xi_sum']         += r['xi_sum']
            agg['gamma_obs_sum']  += r['gamma_obs_sum']
            agg['gamma_obs2_sum'] += r['gamma_obs2_sum']
            agg['total_ll']       += r['total_ll']
            agg['n_utterances']   += r['n_utterances']
        return agg

    # ── M-step ────────────────────────────────────────────────────────────

    def _m_step(self, agg: dict):
        """Update model parameters from aggregated sufficient statistics."""
        gs  = agg['gamma_sum']
        gi  = agg['gamma_init']
        xi  = agg['xi_sum']
        gos = agg['gamma_obs_sum']
        go2 = agg['gamma_obs2_sum']

        # Initial distribution
        self.log_pi = np.log(gi / (gi.sum() + 1e-300) + 1e-300)

        # Transition matrix
        A_new      = xi / (xi.sum(axis=1, keepdims=True) + 1e-300)
        self.log_A = np.log(A_new + 1e-300)

        # Emission means and variances
        for i in range(self.n_states):
            if gs[i] > 1e-8:
                self.means[i] = gos[i] / gs[i]
                self.covs[i]  = go2[i] / gs[i] - self.means[i]**2
                self.covs[i]  = np.maximum(self.covs[i], 1e-6)

    # ── Main training loop ────────────────────────────────────────────────

    def fit(self, utterances: list, verbose: bool = True) -> dict:
        """
        Train using distributed Baum-Welch via Ray.

        Args:
            utterances: list of feature arrays, each shape (T, n_features)
            verbose:    print per-iteration progress

        Returns:
            training log dict
        """
        log_likelihoods  = []
        iter_logs        = []
        comm_times       = []
        prev_ll          = -np.inf

        # Partition data once — reused every iteration
        partitions = self._partition(utterances)

        if verbose:
            print(f"    Workers={self.n_workers}  "
                  f"Utterances={len(utterances)}  "
                  f"Partition sizes={[len(p) for p in partitions]}")

        for iteration in range(self.n_iter):
            iter_start = time.time()

            # Snapshot current model params
            model_params = {
                'log_pi': self.log_pi.copy(),
                'log_A':  self.log_A.copy(),
                'means':  self.means.copy(),
                'covs':   self.covs.copy(),
            }

            # ── Dispatch E-step to Ray workers (all run in parallel) ───
            e_start  = time.time()
            futures  = [
                ray_worker_e_step.remote(wid, model_params, partitions[wid])
                for wid in range(self.n_workers)
            ]
            # ray.get() blocks until all workers finish — this is the
            # synchronisation point (barrier)
            worker_results = ray.get(futures)
            e_time   = time.time() - e_start

            # ── Reduction ─────────────────────────────────────────────
            r_start  = time.time()
            agg      = self._reduce(worker_results)
            r_time   = time.time() - r_start

            # ── M-step ────────────────────────────────────────────────
            m_start  = time.time()
            self._m_step(agg)
            m_time   = time.time() - m_start

            total_ll  = agg['total_ll']
            ll_change = total_ll - prev_ll
            iter_time = time.time() - iter_start

            log_likelihoods.append(total_ll)
            comm_times.append(r_time)

            w_times = [r['elapsed'] for r in worker_results]
            iter_logs.append({
                'iteration':    iteration + 1,
                'total_ll':     total_ll,
                'e_step_time':  e_time,
                'reduce_time':  r_time,
                'mstep_time':   m_time,
                'iter_time':    iter_time,
                'worker_times': w_times,
            })

            if verbose:
                print(f"    Iter {iteration+1:3d}: "
                      f"ll={total_ll:.2f}  (Δ={ll_change:+.4f})  "
                      f"E={e_time:.2f}s  reduce={r_time*1000:.1f}ms  "
                      f"total={iter_time:.2f}s")

            if abs(ll_change) < self.tol and iteration > 0:
                if verbose:
                    print(f"    Converged at iteration {iteration+1}.")
                break

            prev_ll = total_ll

        return {
            'log_likelihoods': log_likelihoods,
            'iter_logs':       iter_logs,
            'comm_times':      comm_times,
            'n_workers':       self.n_workers,
            'n_utterances':    len(utterances),
        }

    def score(self, features: np.ndarray) -> float:
        """Log-likelihood of features under this model."""
        log_B     = _log_emission(features, self.means, self.covs, self.n_states)
        T         = len(features)
        log_alpha = np.zeros((T, self.n_states))
        log_alpha[0] = self.log_pi + log_B[0]
        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = (_log_sum_exp(log_alpha[t-1] + self.log_A[:, j])
                                   + log_B[t, j])
        return float(_log_sum_exp(log_alpha[-1]))

    def get_params(self) -> dict:
        return {
            'log_pi': self.log_pi.copy(),
            'log_A':  self.log_A.copy(),
            'means':  self.means.copy(),
            'covs':   self.covs.copy(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Distributed Speech Recognizer
# ─────────────────────────────────────────────────────────────────────────────

class DistributedSpeechRecognizer:
    """
    Multi-class HMM recognizer with distributed training.
    Trains one DistributedHMMTrainer per phoneme class using Ray.
    Inference is identical to the sequential version.
    """

    def __init__(self, n_states=5, n_mfcc=13, n_workers=4, n_iter=20):
        self.n_states  = n_states
        self.n_mfcc    = n_mfcc
        self.n_workers = n_workers
        self.n_iter    = n_iter
        self.models    = {}
        self.classes   = []
        self.is_fitted = False

    def train(self, data: list, verbose: bool = True) -> dict:
        class_data = {}
        for features, label in data:
            class_data.setdefault(str(label), []).append(features)

        self.classes    = sorted(class_data.keys())
        training_report = {}
        total_start     = time.time()

        print(f"\n{'='*65}")
        print(f"Distributed Training — Ray — {self.n_workers} workers — "
              f"{len(self.classes)} classes")
        print(f"{'='*65}")

        for label in self.classes:
            utterances = class_data[label]
            if verbose:
                print(f"\n  Class '{label}': {len(utterances)} utterances")

            trainer = DistributedHMMTrainer(
                n_states=self.n_states,
                n_features=self.n_mfcc,
                n_workers=min(self.n_workers, max(1, len(utterances))),
                n_iter=self.n_iter,
                seed=hash(label) % 2**31,
            )
            result             = trainer.fit(utterances, verbose=verbose)
            self.models[label] = trainer
            training_report[label] = result

        print(f"\n  Total training time: {time.time()-total_start:.2f}s")
        self.is_fitted = True
        return training_report

    def predict(self, features: np.ndarray) -> tuple:
        if not self.is_fitted:
            raise RuntimeError("Call train() first.")
        scores = {}
        for label, model in self.models.items():
            try:
                scores[label] = model.score(features)
            except Exception:
                scores[label] = -np.inf
        best = max(scores, key=scores.get)
        return best, scores[best], scores

    def evaluate(self, test_data: list) -> dict:
        correct = 0
        times   = []
        for features, true_label in test_data:
            t0 = time.time()
            pred, _, _ = self.predict(features)
            times.append(time.time() - t0)
            if str(pred) == str(true_label):
                correct += 1
        acc   = correct / len(test_data) if test_data else 0
        times = np.array(times) * 1000
        print(f"\n  Accuracy:  {correct}/{len(test_data)} = {acc*100:.1f}%")
        print(f"  Mean decode latency: {times.mean():.2f} ms")
        return {'accuracy': acc, 'correct': correct,
                'total': len(test_data), 'mean_latency_ms': times.mean()}