"""
viterbi.py - Viterbi Decoding for HMM Speech Recognition

The Viterbi algorithm finds the most likely sequence of hidden states
given a sequence of observations. In speech recognition, this gives us
the most probable phoneme sequence for a given audio input.

It uses dynamic programming to efficiently search through all possible
state sequences without evaluating every combination explicitly.
"""


import numpy as np
from hmm import GaussianHMM


def viterbi_decode(model: GaussianHMM, features: np.ndarray) -> tuple:
    """
    Run Viterbi decoding to find the most probable state sequence.

    Algorithm:
        1. delta[t, i] = log P of most likely path ending in state i at time t
        2. psi[t, i]   = which state at t-1 led to the best path ending in i at t
        3. Backtrack from the last frame to recover the full state sequence

    Args:
        model:    Trained GaussianHMM
        features: array of shape (T, n_features)

    Returns:
        state_sequence: list of integers, best state path of length T
        log_prob:       scalar, log probability of the best path
    """
    T = len(features)
    n_states = model.n_states

    # Compute log emission probabilities for all frames
    log_B = model._log_emission(features)  # shape (T, n_states)

    # --- Initialization ---
    delta = np.full((T, n_states), -np.inf)
    psi = np.zeros((T, n_states), dtype=int)  # backpointer matrix

    delta[0] = model.log_pi + log_B[0]

    # --- Recursion ---
    for t in range(1, T):
        for j in range(n_states):
            # For each state j, find which previous state i maximizes the path
            scores = delta[t - 1] + model.log_A[:, j]  # shape (n_states,)
            best_prev = np.argmax(scores)
            delta[t, j] = scores[best_prev] + log_B[t, j]
            psi[t, j] = best_prev

    # --- Termination ---
    log_prob = np.max(delta[-1])
    last_state = np.argmax(delta[-1])

    # --- Backtracking ---
    state_sequence = [0] * T
    state_sequence[-1] = last_state

    for t in range(T - 2, -1, -1):
        state_sequence[t] = psi[t + 1, state_sequence[t + 1]]

    return state_sequence, log_prob


def states_to_labels(state_sequence: list, state_label_map: dict) -> list:
    """
    Convert a sequence of state indices to human-readable labels.

    In a real ASR system, multiple HMM states map to one phoneme.
    This function handles that mapping.

    Args:
        state_sequence:  list of integer state indices
        state_label_map: dict mapping state index -> label string
                         e.g. {0: 'silence', 1: 'vowel', 2: 'consonant'}

    Returns:
        label_sequence: list of label strings
    """
    return [state_label_map.get(s, f"state_{s}") for s in state_sequence]


def compress_state_sequence(state_sequence: list) -> list:
    """
    Compress consecutive repeated states into a single entry.

    Example: [0, 0, 1, 1, 1, 2] -> [(0, 2), (1, 3), (2, 1)]
             meaning: state 0 lasted 2 frames, state 1 lasted 3, state 2 lasted 1.

    Args:
        state_sequence: list of state indices

    Returns:
        compressed: list of (state, duration) tuples
    """
    if not state_sequence:
        return []

    compressed = []
    current_state = state_sequence[0]
    count = 1

    for s in state_sequence[1:]:
        if s == current_state:
            count += 1
        else:
            compressed.append((current_state, count))
            current_state = s
            count = 1

    compressed.append((current_state, count))
    return compressed


def classify_utterance(models: dict, features: np.ndarray) -> tuple:
    """
    Classify an utterance by finding which HMM gives the highest likelihood.

    This is the core of multi-class HMM speech recognition:
    train one HMM per class (e.g., phoneme or word), then pick the best fit.

    Args:
        models:   dict mapping class label -> GaussianHMM
                  e.g. {'silence': hmm0, 'vowel': hmm1, 'consonant': hmm2}
        features: array of shape (T, n_features)

    Returns:
        best_label:    predicted class label (string)
        best_score:    log-likelihood of the best model
        all_scores:    dict mapping label -> log-likelihood score
    """
    all_scores = {}

    for label, model in models.items():
        try:
            score = model.score(features)
            all_scores[label] = score
        except Exception as e:
            all_scores[label] = -np.inf

    best_label = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_label]

    return best_label, best_score, all_scores
