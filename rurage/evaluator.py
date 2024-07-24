import pandas as pd
import numpy as np
import pandas as pd
import torch.nn.functional as F
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from torch import Tensor
from tqdm.auto import tqdm
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer)
from .config import RAGESetConfig, RAGEModelConfig
from .report import RAGEReport
from .metrics import compute_nli_score, compute_similarity, compute_overlap, calculate_rouge, calculate_bleu

class RAGEvaluator:
    def __init__(self, golden_set_cfg: RAGESetConfig) -> None:
        self.golden_set_cfg = golden_set_cfg

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.nli_model_name = None
        self._nli_model = None
        self._nli_tokenizer = None
        self._nli_labels = None

        self.sim_model_name = None
        self._sim_model = None
        self._sim_tokenizer = None

        self._unigram_tokenizer = Tokenizer(
            lower_case=True,
            remove_punctuation=True,
            remove_stopwords=True,
            lemm=True,
            stem=False,
        )
        self._bigram_tokenizer = Tokenizer(
            lower_case=True,
            remove_punctuation=True,
            remove_stopwords=True,
            lemm=True,
            stem=False,
            ngram_range=(2,),
        )
        self._rouge_scorer = rouge_scorer.RougeScorer(
            ["rougeL"], tokenizer=self._unigram_tokenizer
        )

    def _predict_relation(self, context, answer):
        premise = row[prem_col]
        hypothesis = row[hypo_col]
        input = self._nli_tokenizer(
            premise, hypothesis, truncation=True, return_tensors="pt"
        )
        output = self._nli_model(input["input_ids"].to(self.device))
        label = int(torch.argmax(output["logits"]))

        return label

    def _nli_score(self, context, answer):
        return compute_nli_score(self.nli_model, context, answer)

    def _average_pool(self, scores):
        # Implementation of average pooling
        pass

    def _encode_sentences(self, sentences):
        # Implementation of sentence encoding
        pass

    def _predict_similarity(self, context, answer):
        return compute_similarity(context, answer)

    def _compute_overlap(self, reference, candidate):
        return compute_overlap(reference, candidate)

    def _get_overlap_metric_columns(self):
        # Implementation to get overlap metric columns
        pass

    def _calculate_rouge(self, reference, candidate):
        return calculate_rouge(reference, candidate)

    def _calculate_bleu(self, reference, candidate):
        return calculate_bleu(reference, candidate)

    def _init_nli_model(self):
        # Initialize the NLI model
        pass

    def correctness_report(self):
        # Implementation of correctness report
        pass

    def faithfulness_report(self):
        # Implementation of faithfulness report
        pass

    def relevance_report(self):
        # Implementation of relevance report
        pass

    def evaluate_correctness(self):
        # Implementation to evaluate correctness
        pass

    def evaluate_faithfulness(self):
        # Implementation to evaluate faithfulness
        pass

    def evaluate_relevance(self):
        # Implementation to evaluate relevance
        pass

    def comprehensive_evaluation(self):
        # Comprehensive evaluation
        pass

