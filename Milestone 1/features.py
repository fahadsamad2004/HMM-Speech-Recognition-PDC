"""
features.py - MFCC Feature Extraction for HMM Speech Recognition
Converts raw audio files into Mel-Frequency Cepstral Coefficients (MFCCs)
"""


import numpy as np


def extract_mfcc(audio_path: str, n_mfcc: int = 13, sr: int = 16000) -> np.ndarray:
    """
    Extract MFCC features from an audio file.

    Args:
        audio_path: Path to .wav audio file
        n_mfcc:     Number of MFCC coefficients (13 is standard for speech)
        sr:         Target sample rate in Hz

    Returns:
        mfcc: numpy array of shape (n_frames, n_mfcc)
              Each row is one time frame; each column is one MFCC coefficient.
    """
    import librosa
    # Load audio, resample to target sr automatically
    y, _ = librosa.load(audio_path, sr=sr)

    # Compute MFCCs: shape is (n_mfcc, n_frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Transpose to (n_frames, n_mfcc) — time along rows, features along columns
    mfcc = mfcc.T

    return mfcc


def extract_mfcc_with_delta(audio_path: str, n_mfcc: int = 13, sr: int = 16000) -> np.ndarray:
    """
    Extract MFCCs plus delta (velocity) and delta-delta (acceleration) features.
    This gives 3x the features and often improves recognition accuracy.

    Args:
        audio_path: Path to .wav audio file
        n_mfcc:     Number of base MFCC coefficients
        sr:         Target sample rate in Hz

    Returns:
        features: numpy array of shape (n_frames, n_mfcc * 3)
    """
    import librosa
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Delta features capture how MFCCs change over time (1st derivative)
    delta = librosa.feature.delta(mfcc)

    # Delta-delta features (2nd derivative)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack and transpose: shape (n_frames, n_mfcc * 3)
    features = np.concatenate([mfcc, delta, delta2], axis=0).T

    return features


def generate_synthetic_features(n_utterances: int = 10,
                                 n_frames_range: tuple = (50, 150),
                                 n_features: int = 13,
                                 n_classes: int = 3,
                                 seed: int = 42) -> tuple:
    """
    Generate synthetic MFCC-like data for testing without real audio files.
    Each class has a different mean so the HMM has something to learn.

    Args:
        n_utterances:    Number of utterances to generate
        n_frames_range:  (min, max) number of frames per utterance
        n_features:      Feature vector dimensionality
        n_classes:       Number of distinct phoneme classes
        seed:            Random seed for reproducibility

    Returns:
        utterances: list of (features, label) tuples
                    features: array of shape (n_frames, n_features)
                    label:    integer class label
    """
    rng = np.random.RandomState(seed)
    utterances = []

    # Create distinct means for each class so classes are separable
    class_means = [rng.randn(n_features) * 3 for _ in range(n_classes)]

    for i in range(n_utterances):
        label = i % n_classes
        n_frames = rng.randint(*n_frames_range)

        # Generate frames around this class's mean with some noise
        features = class_means[label] + rng.randn(n_frames, n_features)
        utterances.append((features, label))

    return utterances


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance (per feature dimension).
    This is important for numerical stability in the HMM.

    Args:
        features: array of shape (n_frames, n_features)

    Returns:
        normalized: same shape, zero mean and unit variance per column
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8  # avoid division by zero
    return (features - mean) / std
