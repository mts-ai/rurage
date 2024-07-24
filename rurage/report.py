from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class RAGEReport:
    """A report class containing metrics to evaluate RAG model."""

    report_name: str
    entailment_score: Optional[float] = None
    neutral_score: Optional[float] = None
    contradiction_score: Optional[float] = None
    similarity_score: Optional[float] = None
    unigram_overlap_precision: Optional[float] = None
    unigram_overlap_recall: Optional[float] = None
    unigram_overlap_f1: Optional[float] = None
    bigram_overlap_precision: Optional[float] = None
    bigram_overlap_recall: Optional[float] = None
    bigram_overlap_f1: Optional[float] = None
    rouge_precision: Optional[float] = None
    rouge_recall: Optional[float] = None
    rouge_f1: Optional[float] = None
    bleu_score: Optional[float] = None
