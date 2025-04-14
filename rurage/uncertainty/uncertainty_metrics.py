import numpy as np


def mean_token_entropy(entropy: np.ndarray) -> np.ndarray:
    """
    Estimates the sequence-level uncertainty of a language model by calculating the
    mean entropy among all tokens in the generation.

    Parameters:
        entropy (np.ndarray): entropy of probabilities at each token position in the generation.
    Returns:
        np.ndarray: minus log probabilities for each sample.
            Higher values indicate more uncertain samples.
    """
    return np.array([np.mean(e) for e in entropy])


def maximum_sequence_probability(log_likelihoods: np.ndarray) -> np.ndarray:
    """
    Estimates the sequence-level uncertainty of a language model by calculating the
    log-probability of the generation with minus sign.
    It is calculated as the sum of log-probabilities in each token.

    Parameters:
         log_likelihoods (np.ndarray): log likelihoods of generated sequences.
    Returns:
        np.ndarray: minus log probabilities for each sample.
            Higher values indicate more uncertain samples.
    """
    return np.array([-np.sum(log_likelihood) for log_likelihood in log_likelihoods])


def maximum_token_probability(log_likelihoods: np.ndarray) -> np.ndarray:
    """
    Estimates the token-level uncertainty of a language model by calculating the
    log-probability for each token during autoregressive generation.

    Parameters:
        log_likelihoods (np.ndarray): log likelihoods of generated sequences.
    Returns:
        np.ndarray: concatenated minus log probabilities for each token.
            Higher values indicate more uncertain samples.
    """
    return np.array(
        [
            -np.exp(np.array(log_likelihood[:-1]))[0]
            for log_likelihood in log_likelihoods
        ]
    )


def mean_token_probability(log_likelihoods: np.ndarray) -> np.ndarray:
    """
    Estimates the token-level uncertainty of a language model by calculating the mean
    log-probability of all tokens during autoregressive generation.

    Parameters:
        log_likelihoods (np.ndarray): log likelihoods of generated sequences.
    Returns:
        np.ndarray: mean of log probabilities for all tokens.
            Higher values indicate more uncertain samples.
    """
    return np.array(
        [
            np.mean(
                np.concatenate(
                    [
                        -np.exp(np.array(log_likelihood[:-1]))
                        for log_likelihood in log_likelihoods
                    ]
                )
            )
        ]
    )
