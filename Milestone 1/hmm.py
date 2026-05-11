"""
hmm.py - Hidden Markov Model with Baum-Welch Training
Implements a Gaussian-emission HMM for speech recognition.

Key concepts:
- States: Hidden phoneme states (e.g., silence, vowel, consonant)
- Transitions: Probability of moving from state i to state j
- Emissions: Gaussian probability of observing a feature vector in a given state
- Baum-Welch: EM algorithm to learn all parameters from data
"""


import numpy as np
from scipy.stats import multivariate_normal


class GaussianHMM:
    """
    Hidden Markov Model with diagonal Gaussian emissions.

    Parameters are learned via the Baum-Welch (forward-backward) algorithm.
    Diagonal covariance means each feature dimension is treated as independent,
    which is a common and effective simplification for speech.
    """

    def __init__(self, n_states: int, n_features: int, seed: int = 42):
        """
        Initialize HMM with random parameters.

        Args:
            n_states:   Number of hidden states (e.g., 3 states per phoneme)
            n_features: Dimensionality of feature vectors (e.g., 13 for MFCCs)
            seed:       Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_features = n_features
        rng = np.random.RandomState(seed)

        # --- Initial state distribution (pi) ---
        # pi[i] = probability of starting in state i
        pi = rng.dirichlet(np.ones(n_states))
        self.log_pi = np.log(pi + 1e-300)

        # --- Transition matrix (A) ---
        # A[i, j] = probability of moving from state i to state j
        # Each row sums to 1 (it's a probability distribution over next states)
        A = rng.dirichlet(np.ones(n_states), size=n_states)
        self.log_A = np.log(A + 1e-300)

        # --- Emission parameters (Gaussian per state) ---
        # means[i]: mean feature vector for state i
        # covs[i]:  diagonal covariance for state i (stored as variance vector)
        self.means = rng.randn(n_states, n_features)
        self.covs = np.ones((n_states, n_features))  # start with identity covariance

    # -------------------------------------------------------------------------
    # Core probability computation
    # -------------------------------------------------------------------------

    def _log_emission(self, features: np.ndarray) -> np.ndarray:
        """
        Compute log emission probabilities for all frames and states.

        Args:
            features: array of shape (T, n_features) — T time frames

        Returns:
            log_B: array of shape (T, n_states)
                   log_B[t, i] = log P(features[t] | state i)
        """
        T = len(features)
        log_B = np.zeros((T, self.n_states))

        for i in range(self.n_states):
            # Use diagonal Gaussian: log P(x | mu, sigma^2)
            # = -0.5 * sum_d [ (x_d - mu_d)^2 / sigma^2_d + log(2*pi*sigma^2_d) ]
            diff = features - self.means[i]  # shape (T, n_features)
            log_B[:, i] = -0.5 * np.sum(
                diff ** 2 / (self.covs[i] + 1e-8) + np.log(2 * np.pi * (self.covs[i] + 1e-8)),
                axis=1
            )

        return log_B

    # -------------------------------------------------------------------------
    # Forward-Backward Algorithm (E-step of Baum-Welch)
    # -------------------------------------------------------------------------

    def _forward(self, log_B: np.ndarray) -> tuple:
        """
        Forward algorithm in log space to avoid numerical underflow.

        Computes alpha[t, i] = log P(o_1, ..., o_t, q_t = i | model)

        Args:
            log_B: log emission probs, shape (T, n_states)

        Returns:
            log_alpha: shape (T, n_states)
            log_likelihood: scalar, log P(observations | model)
        """
        T = self.n_states
        T_obs = len(log_B)
        log_alpha = np.zeros((T_obs, self.n_states))

        # Initialization: t=0
        log_alpha[0] = self.log_pi + log_B[0]

        # Recursion: t=1,...,T-1
        for t in range(1, T_obs):
            for j in range(self.n_states):
                # log-sum-exp trick for numerical stability
                # alpha[t,j] = sum_i alpha[t-1,i] * A[i,j] * B[j, o_t]
                log_alpha[t, j] = self._log_sum_exp(log_alpha[t - 1] + self.log_A[:, j]) + log_B[t, j]

        log_likelihood = self._log_sum_exp(log_alpha[-1])
        return log_alpha, log_likelihood

    def _backward(self, log_B: np.ndarray) -> np.ndarray:
        """
        Backward algorithm in log space.

        Computes beta[t, i] = log P(o_{t+1}, ..., o_T | q_t = i, model)

        Args:
            log_B: log emission probs, shape (T, n_states)

        Returns:
            log_beta: shape (T, n_states)
        """
        T_obs = len(log_B)
        log_beta = np.zeros((T_obs, self.n_states))

        # Initialization: last time step (beta_T = 1, log_beta_T = 0)
        log_beta[-1] = 0.0

        # Recursion: go backwards
        for t in range(T_obs - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = self._log_sum_exp(
                    self.log_A[i] + log_B[t + 1] + log_beta[t + 1]
                )

        return log_beta

    def _compute_sufficient_statistics(self, features: np.ndarray) -> dict:
        """
        E-step: compute sufficient statistics for one utterance.

        These are the expected counts that drive parameter updates.

        Args:
            features: array of shape (T, n_features)

        Returns:
            stats dict containing:
                gamma:  shape (T, n_states) — state occupation probabilities
                xi:     shape (T-1, n_states, n_states) — transition probabilities
                log_likelihood: scalar
        """
        T = len(features)
        log_B = self._log_emission(features)

        log_alpha, log_likelihood = self._forward(log_B)
        log_beta = self._backward(log_B)

        # Gamma: probability of being in state i at time t
        # gamma[t, i] = P(q_t = i | O, model)
        log_gamma = log_alpha + log_beta
        log_gamma -= self._log_sum_exp(log_gamma, axis=1, keepdims=True)  # normalize
        gamma = np.exp(log_gamma)

        # Xi: probability of transitioning from state i to j at time t
        # xi[t, i, j] = P(q_t=i, q_{t+1}=j | O, model)
        log_xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi[t, i, j] = (log_alpha[t, i] + self.log_A[i, j] +
                                        log_B[t + 1, j] + log_beta[t + 1, j])
            # Normalize xi at each time step
            log_xi[t] -= self._log_sum_exp(log_xi[t].ravel())

        xi = np.exp(log_xi)

        return {
            'gamma': gamma,
            'xi': xi,
            'log_likelihood': log_likelihood,
            'T': T
        }

    # -------------------------------------------------------------------------
    # Baum-Welch Training (M-step)
    # -------------------------------------------------------------------------

    def fit(self, utterances: list, n_iter: int = 20, tol: float = 1e-4) -> list:
        """
        Train HMM parameters using the Baum-Welch algorithm.

        Iterates E-step (compute statistics) and M-step (update parameters)
        until convergence or max iterations.

        Args:
            utterances: list of feature arrays, each shape (T_i, n_features)
            n_iter:     maximum number of EM iterations
            tol:        convergence threshold on log-likelihood change

        Returns:
            log_likelihoods: list of total log-likelihood per iteration
        """
        log_likelihoods = []
        prev_ll = -np.inf

        for iteration in range(n_iter):
            # --- E-step: collect sufficient statistics over all utterances ---
            total_ll = 0.0

            # Accumulators for M-step
            gamma_sum = np.zeros(self.n_states)           # sum of gamma over time
            gamma_init = np.zeros(self.n_states)           # gamma at t=0
            xi_sum = np.zeros((self.n_states, self.n_states))  # sum of xi over time
            gamma_obs_sum = np.zeros((self.n_states, self.n_features))  # weighted obs sum
            gamma_obs2_sum = np.zeros((self.n_states, self.n_features)) # weighted obs^2 sum

            for features in utterances:
                stats = self._compute_sufficient_statistics(features)
                gamma = stats['gamma']
                xi = stats['xi']

                total_ll += stats['log_likelihood']

                # Accumulate statistics across utterances
                gamma_sum += gamma.sum(axis=0)
                gamma_init += gamma[0]
                xi_sum += xi.sum(axis=0)

                # Weighted feature sums for mean/covariance update
                for i in range(self.n_states):
                    gamma_obs_sum[i] += (gamma[:, i:i+1] * features).sum(axis=0)
                    gamma_obs2_sum[i] += (gamma[:, i:i+1] * features ** 2).sum(axis=0)

            log_likelihoods.append(total_ll)

            # --- M-step: update parameters ---

            # Update initial state distribution
            self.log_pi = np.log(gamma_init / gamma_init.sum() + 1e-300)

            # Update transition matrix (normalize each row)
            A_new = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)
            self.log_A = np.log(A_new + 1e-300)

            # Update emission means and covariances
            for i in range(self.n_states):
                if gamma_sum[i] > 1e-8:
                    self.means[i] = gamma_obs_sum[i] / gamma_sum[i]
                    # Variance = E[x^2] - E[x]^2
                    self.covs[i] = (gamma_obs2_sum[i] / gamma_sum[i]) - self.means[i] ** 2
                    self.covs[i] = np.maximum(self.covs[i], 1e-6)  # floor to avoid zero variance

            # Check convergence
            ll_change = total_ll - prev_ll
            print(f"  Iteration {iteration + 1:3d}: log-likelihood = {total_ll:.4f}  (Δ = {ll_change:+.4f})")

            if abs(ll_change) < tol and iteration > 0:
                print(f"  Converged at iteration {iteration + 1}.")
                break

            prev_ll = total_ll

        return log_likelihoods

    def score(self, features: np.ndarray) -> float:
        """
        Compute log-likelihood of an observation sequence given this model.

        Args:
            features: array of shape (T, n_features)

        Returns:
            log_likelihood: scalar
        """
        log_B = self._log_emission(features)
        _, log_likelihood = self._forward(log_B)
        return log_likelihood

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    @staticmethod
    def _log_sum_exp(a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
        """
        Numerically stable log-sum-exp.
        log(sum(exp(a))) computed without overflow.
        """
        a_max = np.max(a, axis=axis, keepdims=True)
        result = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result

    def __repr__(self):
        return f"GaussianHMM(n_states={self.n_states}, n_features={self.n_features})"
