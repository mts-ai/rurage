from typing import Dict, Tuple

import nltk
import pandas as pd
import rouge_score

from .tokenizer import Tokenizer


def compute_nli_score(
    labels: pd.Series,
    mtype: str,
    class_labels: Dict,
    norm_size: int,
) -> float:
    counts = labels.value_counts()
    for id, label in class_labels.items():
        if label == mtype:
            try:
                return counts[int(id)] / norm_size
            except KeyError:
                return 0.0


def compute_similarity(embedding_first, embedding_second) -> float:
    return embedding_first @ embedding_second.T


def compute_overlap(reference_sentence, sentence, tokenizer: Tokenizer) -> Tuple[float]:
    set_first = set(tokenizer(reference_sentence))
    set_second = set(tokenizer(sentence))

    intersection = len(set_first.intersection(set_second))
    union = len(set_first.union(set_second))

    try:
        precision = intersection / len(set_first)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = intersection / union
    except ZeroDivisionError:
        recall = 0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0

    return precision, recall, f1


def calculate_rouge(reference, candidate, rouge_scorer) -> "rouge_score.scoring.Score":
    scores = rouge_scorer.score(target=reference, prediction=candidate)
    return scores["rougeL"]


def calculate_bleu(reference, candidate) -> float:
    bleu = nltk.translate.bleu_score.sentence_bleu([reference], candidate, weights=(1,))
    return bleu
