from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .llm_generation import LLMConfig, generate
from .uncertainty_metrics import (
    maximum_sequence_probability,
    maximum_token_probability,
    mean_token_entropy,
    mean_token_probability,
)
from .uncertainty_stats import entropy_calculator, greedy_probs_calculator


def estimate_uncertainty(
    input_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_config: GenerationConfig = None,
    model_path_or_repo_id: str = None,
    prompt: str = None,
) -> dict[float, float, float, float, str]:
    """
    Estimate uncertainty metrics for a LLM's text generation.

    Calculates various uncertainty metrics for a given input text using a language model,
    including mean token entropy, token probabilities, and sequence probabilities.

    Args:
        input_text (str): text to generate uncertainty metrics for.
        model (AutoModelForCausalLM): LLM to use for text generation.
        tokenizer (AutoTokenizer): corresponding tokenizer for the model.
        generation_config (GenerationConfig, optional): configuration for text generation. Defaults to None.
        model_path_or_repo_id (str, optional): path or repository id to load generation config. Defaults to None.
        prompt (str, optional): optional prompt to modify generation behavior. Defaults to None.

    Returns:
        dict[float, float, float, float, str]: dictionary containing uncertainty metrics:
            - 'MeanTokenEntropy': average entropy of token probabilities
            - 'MeanTokenProbability': average token probability
            - 'MaxTokenProbability': maximum token probability
            - 'MaxSequenceProbability': maximum sequence probability
            - 'Output': generated text output
    """
    if not generation_config:
        if not model_path_or_repo_id:
            raise ValueError(
                "Provide generation_config or model_path_or_repo_id where generation config is stored."
            )

        try:
            generation_config = GenerationConfig.from_pretrained(model_path_or_repo_id)
        except OSError:
            raise ValueError(
                "There is no generation config in the provided repository."
            )

    llm_config = LLMConfig(
        model=model, tokenizer=tokenizer, generation_config=generation_config
    )
    tokenized_texts, model_output = generate(llm_config, [input_text], prompt)

    greedy_probs_stats = greedy_probs_calculator(
        tokenized_texts, model_output, llm_config
    )
    entropy_stats = entropy_calculator(greedy_probs_stats["log_probs"])

    mean_token_entropy_metric = mean_token_entropy(entropy_stats["entropy"])

    mean_token_prob_metric = mean_token_probability(
        greedy_probs_stats["log_likelihoods"]
    )

    max_token_prob_metric = maximum_token_probability(
        greedy_probs_stats["log_likelihoods"]
    )

    max_seq_prob_metric = maximum_sequence_probability(
        greedy_probs_stats["log_likelihoods"]
    )

    ue_metrics = {
        "MeanTokenEntropy": mean_token_entropy_metric[0],
        "MeanTokenProbability": mean_token_prob_metric[0],
        "MaxTokenProbability": max_token_prob_metric[0],
        "MaxSequenceProbability": max_seq_prob_metric[0],
        "Output": greedy_probs_stats["outputs"][0],
    }
    return ue_metrics
