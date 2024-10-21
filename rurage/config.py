from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class RAGEModelConfig:
    """A config to define information about RAG model answer."""

    context_col: str
    answer_col: str


@dataclass
class RAGESetConfig:
    """A config to define information about evaluation set structure."""

    golden_set: pd.DataFrame
    question_col: str
    golden_answer_col: str
    models_cfg: List[RAGEModelConfig]

    def __post_init__(self):
        object.__setattr__(
            self, "golden_set", self.golden_set.copy().reset_index(drop=True)
        )


ensemble_features = {
    "correctness": [
        "nli",
        "sim",
        "unigram_overlap_precision",
        "unigram_overlap_recall",
        "unigram_overlap_f1",
        "bigram_overlap_precision",
        "bigram_overlap_recall",
        "bigram_overlap_f1",
        "rouge_precision",
        "rouge_recall",
        "rouge_f1",
        "bleu",
    ],
    "faithfulness": [
        "nli",
        "unigram_overlap_precision",
        "rouge_precision",
    ],
    "relevance": [
        "sim",
        "unigram_overlap_precision",
        "rouge_precision",
    ],
}
