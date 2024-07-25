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
        object.__setattr__(self, "golden_set", self.golden_set.copy())
