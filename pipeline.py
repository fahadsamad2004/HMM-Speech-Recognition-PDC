"""
pipeline.py - End-to-End HMM Speech Recognition Pipeline

This module ties together feature extraction, HMM training, and Viterbi
decoding into a single SpeechRecognizer class. One HMM is trained per class
(phoneme or word), and recognition is done by finding the best-scoring model.
"""


import numpy as np
import time
from features import extract_mfcc, normalize_features
from hmm import GaussianHMM
from viterbi import classify_utterance, viterbi_decode, compress_state_sequence


class SpeechRecognizer:
    """
    Multi-class HMM-based speech recognizer.

    Workflow:
        1. train(data)     — fit one HMM per class on labelled utterances
        2. predict(audio)  — decode a new utterance, return predicted label
        3. evaluate(data)  — compute accuracy on a labelled test set
    """

    def __init__(self, n_states: int = 5, n_mfcc: int = 13,
                 n_iter: int = 20, sr: int = 16000):
        """
        Args:
            n_states: Number of HMM states per class
            n_mfcc:   Number of MFCC coefficients to extract
            n_iter:   Max Baum-Welch iterations per model
            sr:       Audio sample rate
        """
        self.n_states = n_states
        self.n_mfcc = n_mfcc
        self.n_iter = n_iter
        self.sr = sr
        self.models = {}       # label -> GaussianHMM
        self.classes = []      # list of class labels seen during training
        self.is_fitted = False

    def train(self, data: list, verbose: bool = True) -> None:
        """
        Train one HMM per class using labelled utterances.

        Args:
            data: list of (features, label) tuples
                  features: array of shape (T, n_mfcc)
                  label:    string or int class label
            verbose: whether to print training progress
        """
        # Group utterances by class
        class_data = {}
        for features, label in data:
            label = str(label)
            if label not in class_data:
                class_data[label] = []
            class_data[label].append(features)

        self.classes = sorted(class_data.keys())

        print(f"\n{'='*60}")
        print(f"Training {len(self.classes)} HMM(s) with {self.n_states} states each")
        print(f"{'='*60}")

        total_start = time.time()

        for label in self.classes:
            utterances = class_data[label]
            if verbose:
                print(f"\nClass '{label}': {len(utterances)} training utterance(s)")

            # Create and train one HMM for this class
            model = GaussianHMM(
                n_states=self.n_states,
                n_features=self.n_mfcc,
                seed=hash(label) % 2**31
            )

            start = time.time()
            model.fit(utterances, n_iter=self.n_iter)
            elapsed = time.time() - start

            self.models[label] = model

            if verbose:
                print(f"  Training time: {elapsed:.2f}s")

        total_elapsed = time.time() - total_start
        print(f"\nTotal training time: {total_elapsed:.2f}s")
        self.is_fitted = True

    def predict_features(self, features: np.ndarray) -> tuple:
        """
        Predict the class of a pre-computed feature array.

        Args:
            features: array of shape (T, n_mfcc)

        Returns:
            predicted_label: string class label
            confidence:      log-likelihood of best model
            all_scores:      dict of all model scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        predicted_label, confidence, all_scores = classify_utterance(
            self.models, features
        )
        return predicted_label, confidence, all_scores

    def predict_audio(self, audio_path: str) -> tuple:
        """
        Full pipeline: audio file -> predicted label.

        Args:
            audio_path: path to .wav file

        Returns:
            predicted_label: string class label
            state_sequence:  Viterbi state path for the best-scoring model
            confidence:      log-likelihood score
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        # Step 1: Extract features
        features = extract_mfcc(audio_path, n_mfcc=self.n_mfcc, sr=self.sr)
        features = normalize_features(features)

        # Step 2: Classify
        predicted_label, confidence, _ = classify_utterance(self.models, features)

        # Step 3: Viterbi decode with the best model
        best_model = self.models[predicted_label]
        state_sequence, _ = viterbi_decode(best_model, features)

        return predicted_label, state_sequence, confidence

    def evaluate(self, test_data: list, verbose: bool = True) -> dict:
        """
        Evaluate recognition accuracy on labelled test data.

        Args:
            test_data: list of (features, true_label) tuples
            verbose:   whether to print per-sample results

        Returns:
            results dict containing accuracy, per-class stats, timing
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        correct = 0
        total = len(test_data)
        per_class = {c: {'correct': 0, 'total': 0} for c in self.classes}
        decode_times = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating on {total} test utterance(s)")
            print(f"{'='*60}")
            print(f"{'Sample':>6}  {'True':>12}  {'Predicted':>12}  {'Correct':>7}  {'Time(ms)':>8}")
            print("-" * 60)

        for idx, (features, true_label) in enumerate(test_data):
            true_label = str(true_label)

            start = time.time()
            predicted_label, confidence, _ = self.predict_features(features)
            elapsed_ms = (time.time() - start) * 1000
            decode_times.append(elapsed_ms)

            is_correct = (predicted_label == true_label)
            if is_correct:
                correct += 1

            if true_label in per_class:
                per_class[true_label]['total'] += 1
                if is_correct:
                    per_class[true_label]['correct'] += 1

            if verbose:
                marker = "✓" if is_correct else "✗"
                print(f"{idx+1:>6}  {true_label:>12}  {predicted_label:>12}  {marker:>7}  {elapsed_ms:>8.1f}")

        accuracy = correct / total if total > 0 else 0.0
        avg_time = np.mean(decode_times)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Overall Accuracy: {correct}/{total} = {accuracy * 100:.1f}%")
            print(f"Avg decode time:  {avg_time:.1f} ms per utterance")
            print(f"\nPer-class breakdown:")
            for c in self.classes:
                c_total = per_class[c]['total']
                c_correct = per_class[c]['correct']
                c_acc = (c_correct / c_total * 100) if c_total > 0 else 0
                print(f"  Class '{c}': {c_correct}/{c_total} ({c_acc:.1f}%)")

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_class': per_class,
            'avg_decode_time_ms': avg_time,
            'decode_times_ms': decode_times
        }

    def decode_state_path(self, features: np.ndarray, label: str) -> list:
        """
        Run Viterbi on a specific model and return compressed state path.

        Useful for inspecting which phoneme states were active when.

        Args:
            features: array of shape (T, n_mfcc)
            label:    which class model to use

        Returns:
            compressed: list of (state, duration_frames) tuples
        """
        if label not in self.models:
            raise ValueError(f"Unknown label '{label}'. Known: {list(self.models.keys())}")

        state_seq, log_prob = viterbi_decode(self.models[label], features)
        compressed = compress_state_sequence(state_seq)

        print(f"\nViterbi path for model '{label}' (log prob = {log_prob:.2f}):")
        for state, duration in compressed:
            print(f"  State {state}: {duration} frames")

        return compressed
