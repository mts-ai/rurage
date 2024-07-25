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


def correctness_report(report: RAGEReport) -> None:
    """Prints model evaluation report on correctness task.
    It contains NLI, similarity, uni-/bi-gram overlap (P/R/F1), ROUGE-L (P/R/F1) and BLEU scores.

    Args:
        report (RAGEReport): a model report with calculated metrics.
    """
    print(f"[{report.report_name}]")
    print("\tEntailment score:", report.entailment_score)
    print("\tNeutral score:", report.neutral_score)
    print("\tContradiction score:", report.contradiction_score)
    print("\n")

    print("\tSimilarity score:", report.similarity_score)
    print("\n")

    print("\tToken overlap (1-gram) precision:", report.unigram_overlap_precision)
    print("\tToken overlap (1-gram) recall:", report.unigram_overlap_recall)
    print("\tToken overlap (1-gram) F1:", report.unigram_overlap_f1)
    print("\n")

    print("\tToken overlap (2-gram) precision:", report.bigram_overlap_precision)
    print("\tToken overlap (2-gram) recall:", report.bigram_overlap_recall)
    print("\tToken overlap (12-gram) F1:", report.bigram_overlap_f1)
    print("\n")

    print("\tROUGE-L precision:", report.rouge_precision)
    print("\tROUGE-L recall:", report.rouge_recall)
    print("\tROUGE-L F1:", report.rouge_f1)
    print("\n")

    print("\tBLEU:", report.bleu_score)
    print("\n\n")


def faithfulness_report(report: RAGEReport) -> None:
    """Prints model evaluation report on faithfulness task.
    It contains NLI, unigram overlap (P/R/F1) and ROUGE-L (reversed P) scores.

    Args:
        report (RAGEReport): a model report with calculated metrics.
    """
    print(f"[{report.report_name}]")
    print("\tEntailment score:", report.entailment_score)
    print("\tNeutral score:", report.neutral_score)
    print("\tContradiction score:", report.contradiction_score)
    print("\n")

    print(
        "\tToken overlap (1-gram) reversed precision:", report.unigram_overlap_precision
    )
    print("\n")

    print("\tROUGE-L precision:", report.rouge_precision)
    print("\n\n")


def relevance_report(report: RAGEReport) -> None:
    """Prints model evaluation report on relevance task.
    It contains similarity, unigram overlap (P) and ROUGE-L (R) scores.

    Args:
        report (RAGEReport): a model report with calculated metrics.
    """
    print(f"[{report.report_name}]")
    print("\tSimilarity score:", report.similarity_score)
    print("\n")

    print("\tToken overlap (1-gram) precision:", report.unigram_overlap_precision)
    print("\n")

    print("\tROUGE-L recall:", report.rouge_recall)
    print("\n\n")
