import numpy as np
import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput

from .llm_generation import LLMConfig


def greedy_probs_calculator(
    tokenized_texts: dict,
    model_output: GenerateDecoderOnlyOutput,
    llm_config: LLMConfig,
) -> dict[str, np.ndarray]:
    """
    Calculates the statistics of probabilities at each token position in the generation.

    Args:
        tokenized_texts (dict): input tokenized texts.
        model_output (GenerateDecoderOnlyOutput): model generation output.
        llm_config (LLMConfig): language model configuration.

    Returns:
        dict[str, np.ndarray]: Dictionary containing:
            - 'log_probs': log probabilities for each token.
            - 'outputs': decoded generated texts.
            - 'log_likelihoods': log likelihoods of generated sequences.
    """
    logits = torch.stack(model_output.scores, dim=1)
    sequences = model_output.sequences

    log_probs = []
    tokens = []
    texts = []
    for i in range(len(tokenized_texts["input_ids"])):
        idx = tokenized_texts["input_ids"].shape[1]
        seq = sequences[i, idx:].cpu()

        length, text_length = len(seq), len(seq)
        for j in range(len(seq)):
            if seq[j] == llm_config.tokenizer.eos_token_id:
                length = j + 1
                text_length = j
                break
        tokens.append(seq[:length].tolist())
        texts.append(llm_config.tokenizer.decode(seq[:text_length]))
        log_probs.append(logits[i, :length, :].cpu().numpy())

    log_likelihoods = []
    for i in range(len(tokenized_texts["input_ids"])):
        seq_log_probs = log_probs[i]
        seq_tokens = tokens[i]
        assert len(seq_tokens) == len(seq_log_probs)
        log_likelihoods.append(
            [seq_log_probs[j, seq_tokens[j]] for j in range(len(seq_log_probs))]
        )

    return {
        "log_probs": log_probs,
        "outputs": texts,
        "log_likelihoods": log_likelihoods,
    }


def entropy_calculator(
    log_probs: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Calculates the entropy of probabilities at each token position in the generation.

    Args:
        log_probs (np.ndarray): log probabilities for each token.

    Returns:
        dict[str, np.ndarray]: a list of entropy values for each token position.
    """
    entropies = []
    for seq_log_probs in log_probs:
        entropies.append([])
        for lp in seq_log_probs:
            mask = ~np.isinf(lp)
            entropies[-1].append(-np.sum(np.array(lp[mask]) * np.exp(lp[mask])))
    return {"entropy": entropies}
