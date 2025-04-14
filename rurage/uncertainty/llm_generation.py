from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput


@dataclass
class LLMConfig:
    """A class for storing a model with its generation parameres and tokenizer."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    generation_config: GenerationConfig


class _ScoresProcessor:
    """Stores original token scores instead of the ones modified with generation parameters."""

    def __init__(self):
        self.scores = []

    def __call__(self, input_ids=None, scores=None):
        self.scores.append(scores.log_softmax(-1))
        return scores


def generate(
    llm_config: LLMConfig, query: str, prompt: str
) -> tuple[dict, GenerateDecoderOnlyOutput]:
    """
    Generates text with optional system prompt.

    Args:
        llm_config (LLMConfig): Configuration for the language model, including model, tokenizer,
        and generation settings.
        query (str): The user's input query to generate text for.
        prompt (str): A system-level prompt to guide the model's generation.

    Returns:
        tuple[dict, GenerateDecoderOnlyOutput]: A tuple containing the tokenized input data
        and the model's generation output.
    """
    messages = [{"role": "system", "content": prompt}] if prompt else []
    messages.append({"role": "user", "content": query})

    chat_template = llm_config.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    data = llm_config.tokenizer(
        chat_template, return_tensors="pt", add_special_tokens=False
    )
    data = {k: v.to(llm_config.model.device) for k, v in data.items()}

    processor = _ScoresProcessor()
    logits_processor = LogitsProcessorList([processor])
    output = llm_config.model.generate(
        **data,
        generation_config=llm_config.generation_config,
        min_new_tokens=2,
        output_scores=True,
        return_dict_in_generate=True,
        output_hidden_states=True,
        num_return_sequences=1,
        logits_processor=logits_processor,
    )
    output.scores = processor.scores

    return data, output
