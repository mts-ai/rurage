import re
import string
import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import pymorphy2
import rouge_score
import torch
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

warnings.filterwarnings("ignore")


class Tokenizer:
    def __init__(
        self,
        lower_case: bool,
        remove_punctuation: bool,
        remove_stopwords: bool,
        lemm: bool,
        stem: bool,
        language: Literal["russian", "english"] = "russian",
        ngram_range: Tuple = (1, 1),
    ):
        self.nltk_setup()

        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemm = lemm
        self.stem = stem
        self.language = language
        self.ngram_range = ngram_range
        self._stemmer = SnowballStemmer(language)
        self._lemmatizer = pymorphy2.MorphAnalyzer()
        self._stop_words = set(stopwords.words(language))
        self._punctuation = set(string.punctuation)

        if self.stem and not self.lower_case:
            raise ValueError(
                "Stemming applying lower case to tokens, so lower_case must be True if stem is True"
            )

    @staticmethod
    def nltk_setup() -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

    @staticmethod
    def remove_punct(text: str) -> str:
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        if self.remove_punctuation:
            text = self.remove_punct(text)

        tokens = word_tokenize(text)

        if self.lower_case:
            tokens = [word.lower() for word in tokens]

        if self.remove_punctuation:
            tokens = [word for word in tokens if word not in self._punctuation]

        if self.remove_stopwords:
            if self.lower_case:
                tokens = [word for word in tokens if word not in self._stop_words]
            else:
                tokens = [
                    word for word in tokens if word.lower() not in self._stop_words
                ]

        if self.lemm:
            tokens = [self._lemmatizer.parse(word)[0].normal_form for word in tokens]

        if self.stem:
            tokens = [self._stemmer.stem(word) for word in tokens]

        combined_ngrams = []
        for ngram in range(self.ngram_range[0], self.ngram_range[-1] + 1):
            ngram_tokens = [*ngrams(tokens, ngram)]
            ngram_tokens = [" ".join(item) for item in ngram_tokens]
            combined_ngrams += ngram_tokens

        return combined_ngrams

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)


class Color:
    """A class to provide pretty report."""

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    @staticmethod
    def stylify(text, style):
        return eval(f"Color.{style}") + str(text) + Color.END


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

    # NLI block
    def _predict_relation(self, row: pd.Series, prem_col: str, hypo_col: str) -> float:
        premise = row[prem_col]
        hypothesis = row[hypo_col]
        input = self._nli_tokenizer(
            premise, hypothesis, truncation=True, return_tensors="pt"
        )
        output = self._nli_model(input["input_ids"].to(self.device))
        label = int(torch.argmax(output["logits"]))

        return label

    def _nli_score(self, labels: pd.Series, mtype: str) -> float:
        counts = labels.value_counts()
        norm = self.golden_set_cfg.golden_set.shape[0]
        for id, label in self._nli_labels.items():
            if label == mtype:
                return counts[int(id)] / norm

    # Similarity block
    @staticmethod
    def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _encode_sentences(
        self,
        sentences: np.ndarray,
        batch_size: int = 16,
        prefix: Literal["query", "passage"] = "query",
    ) -> np.ndarray:
        sentences = prefix + ":" + sentences
        batches = np.array_split(
            sentences,
            np.floor(len(sentences) / batch_size),
        )
        all_embeddings = []
        for sentences_batch in batches:
            sentences_batch = list(sentences_batch)
            batch_dict = self._sim_tokenizer(
                sentences_batch,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                embeddings = self._sim_model(**batch_dict)
                embeddings = self._average_pool(
                    embeddings.last_hidden_state, batch_dict["attention_mask"]
                )
                embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()

                all_embeddings.extend(embeddings)

        return all_embeddings

    @staticmethod
    def _predict_similarity(row: pd.Series, ref_col: str, col: str) -> float:
        return row[ref_col] @ row[col].T

    # Token overlap block
    def _compute_overlap(
        self, row: pd.Series, ref_col: str, col: str, ngram: str
    ) -> Tuple[float]:
        if ngram == "unigram":
            tokenizer = self._unigram_tokenizer
        else:
            tokenizer = self._bigram_tokenizer
        set_1 = set(tokenizer(row[ref_col]))
        set_2 = set(tokenizer(row[col]))

        intersection = len(set_1.intersection(set_2))
        union = len(set_1.union(set_2))

        prec = intersection / len(set_1)
        rec = intersection / union
        try:
            f1 = 2 * prec * rec / (prec + rec)
        except ZeroDivisionError:
            f1 = 0
        return prec, rec, f1

    def _get_overlap_metric_columns(
        self, df: pd.DataFrame, col_name: str, ref_col: str, col: str, ngram: str
    ) -> pd.DataFrame:
        overlap_metrics = df.apply(
            lambda row: self._compute_overlap(
                row, ref_col=ref_col, col=col, ngram=ngram
            ),
            axis=1,
        )
        overlap_metrics = pd.DataFrame(
            overlap_metrics.to_list(),
            columns=[
                f"{col_name}_{ngram}_overlap_precision",
                f"{col_name}_{ngram}_overlap_recall",
                f"{col_name}_{ngram}_overlap_f1",
            ],
        )

        return overlap_metrics

    # ROUGE-L block
    def _calculate_rouge(
        self, row: pd.Series, ref_col: str, col: str
    ) -> "rouge_score.scoring.Score":
        reference = row[ref_col]
        candidate = row[col]
        scores = self._rouge_scorer.score(target=reference, prediction=candidate)
        return scores["rougeL"]

    # BLEU block
    def _calculate_bleu(self, row: pd.Series, ref_col: str, col: str) -> float:
        reference = self._unigram_tokenizer(row[ref_col])
        candidate = self._unigram_tokenizer(row[col])

        bleu = nltk.translate.bleu_score.sentence_bleu(
            [reference], candidate, weights=(1,)
        )
        return bleu

    def _init_nli_model(self, model_name: str) -> None:
        if model_name != self.nli_model_name:
            print(f"Initializing the NLI model: {model_name}")
            self.nli_model_name = model_name
            self._nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(
                self.nli_model_name
            ).to(self.device)

            if self.nli_model_name == "MTS-AI-SearchSkill/DeBERTa-nli-ru":
                premise = "Я люблю пиццу"
                hypothesis = "Мне нравится пицца"

                input = self._nli_tokenizer(
                    premise, hypothesis, truncation=True, return_tensors="pt"
                )
                output = self._nli_model(input["input_ids"].to(self.device))
                self._nli_labels = self._nli_model.config.id2label
                pred_class = int(torch.argmax(output["logits"]))
                if self._nli_labels[pred_class] != "entailment":
                    raise AssertionError("The NLI model is not working properly!")
            else:
                print("The NLI model was loaded without basic testing!")
        else:
            print("The NLI model has alredy been loaded.")

    def _init_sim_model(self, model_name: str) -> None:
        if model_name != self.sim_model_name:
            print(f"Initializing the similarity model: {model_name}")
            self.sim_model_name = model_name
            self._sim_tokenizer = AutoTokenizer.from_pretrained(self.sim_model_name)
            self._sim_model = AutoModel.from_pretrained(self.sim_model_name).to(
                self.device
            )

            if self.sim_model_name == "intfloat/multilingual-e5-large":
                sentence_1 = "Пушкин написал много произведений"
                sentence_2 = "А. С. Пушкин сочинил множество стихов и повестей"

                batch_dict = self._sim_tokenizer(
                    [sentence_1, sentence_2],
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self._sim_model(**batch_dict)
                embeddings = self._average_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)

                if (embeddings[0] @ embeddings[1].T).item() <= 0.934:
                    raise AssertionError(
                        "The similarity model is not working properly!"
                    )
            else:
                print("The similarity model was loaded without basic testing!")
        else:
            print("The similarity model has alredy been loaded.")

    def correctness_report(self, report: RAGEReport) -> None:
        """Prints model evaluation report on correctness task.
        It contains NLI, similarity, uni-/bi-gram overlap (P/R/F1), ROUGE-L (P/R/F1) and BLEU scores.

        Args:
            report (RAGEReport): a model report with calculated metrics.
        """
        print(Color.stylify(f"[{report.report_name}]", "GREEN"))
        print("\tEntailment score:", Color.stylify(report.entailment_score, "BOLD"))
        print("\tNeutral score:", Color.stylify(report.neutral_score, "BOLD"))
        print(
            "\tContradiction score:", Color.stylify(report.contradiction_score, "BOLD")
        )
        print("\n")

        print("\tSimilarity score:", Color.stylify(report.similarity_score, "BOLD"))
        print("\n")

        print(
            "\tToken overlap (1-gram) precision:",
            Color.stylify(report.unigram_overlap_precision, "BOLD"),
        )
        print(
            "\tToken overlap (1-gram) recall:",
            Color.stylify(report.unigram_overlap_recall, "BOLD"),
        )
        print(
            "\tToken overlap (1-gram) F1:",
            Color.stylify(report.unigram_overlap_f1, "BOLD"),
        )
        print("\n")

        print(
            "\tToken overlap (2-gram) precision:",
            Color.stylify(report.bigram_overlap_precision, "BOLD"),
        )
        print(
            "\tToken overlap (2-gram) recall:",
            Color.stylify(report.bigram_overlap_recall, "BOLD"),
        )
        print(
            "\tToken overlap (12-gram) F1:",
            Color.stylify(report.bigram_overlap_f1, "BOLD"),
        )
        print("\n")

        print("\tROUGE-L precision:", Color.stylify(report.rouge_precision, "BOLD"))
        print("\tROUGE-L recall:", Color.stylify(report.rouge_recall, "BOLD"))
        print("\tROUGE-L F1:", Color.stylify(report.rouge_f1, "BOLD"))
        print("\n")

        print("\tBLEU:", Color.stylify(report.bleu_score, "BOLD"))
        print("\n\n")

    def faithfulness_report(self, report: RAGEReport):
        """Prints model evaluation report on faithfulness task.
        It contains NLI, unigram overlap (P/R/F1) and ROUGE-L (reversed P) scores.

        Args:
            report (RAGEReport): a model report with calculated metrics.
        """
        print(Color.stylify(f"[{report.report_name}]", "GREEN"))
        print("\tEntailment score:", Color.stylify(report.entailment_score, "BOLD"))
        print("\tNeutral score:", Color.stylify(report.neutral_score, "BOLD"))
        print(
            "\tContradiction score:", Color.stylify(report.contradiction_score, "BOLD")
        )
        print("\n")

        print(
            "\tToken overlap (1-gram) reversed precision:",
            Color.stylify(report.unigram_overlap_precision, "BOLD"),
        )
        print("\n")

        print("\tROUGE-L precision:", Color.stylify(report.rouge_precision, "BOLD"))
        print("\n\n")

    def relevance_report(self, report: RAGEReport):
        """Prints model evaluation report on relevance task.
        It contains similarity, unigram overlap (P) and ROUGE-L (R) scores.

        Args:
            report (RAGEReport): a model report with calculated metrics.
        """
        print(Color.stylify(f"[{report.report_name}]", "GREEN"))
        print("\tSimilarity score:", Color.stylify(report.similarity_score, "BOLD"))
        print("\n")

        print(
            "\tToken overlap (1-gram) precision:",
            Color.stylify(report.unigram_overlap_precision, "BOLD"),
        )
        print("\n")

        print("\tROUGE-L recall:", Color.stylify(report.rouge_recall, "BOLD"))
        print("\n\n")

    def evaluate_correctness(
        self,
        nli_model_name: str = "MTS-AI-SearchSkill/DeBERTa-nli-ru",
        sim_model_name: str = "intfloat/multilingual-e5-large",
        print_report: bool = False,
        pointwise_report: bool = False,
    ) -> List[RAGEReport]:
        """Evaluate models on the correctness task (A~A*).
        It estimates NLI, similarity, uni-/bi-gram overlap (P/R/F1), ROUGE-L (P/R/F1) and BLEU scores.

        Args:
            nli_model_name (str, optional): HF model name to use for the NLI score.
            Defaults to "MTS-AI-SearchSkill/DeBERTa-nli-ru".
            sim_model_name (str, optional): HF model name to use for the Similarity score.
            Defaults to "intfloat/multilingual-e5-large".
            print_report (bool, optional): Whether to print the output to the console.
            Defaults to True.

        Returns:
            List[RAGEReport]: A list of the reports for each model.
        """
        print("Starting correctness evaluation")
        # init NLI
        self._init_nli_model(model_name=nli_model_name)

        # init Similarity
        self._init_sim_model(model_name=sim_model_name)
        # encode golden answer
        self.golden_set_cfg.golden_set[
            f"{self.golden_set_cfg.golden_answer_col}_embs"
        ] = self._encode_sentences(
            self.golden_set_cfg.golden_set[self.golden_set_cfg.golden_answer_col].values
        )
        for model_cfg in tqdm(self.golden_set_cfg.models_cfg, desc="Model #"):
            # NLI
            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"] = (
                self.golden_set_cfg.golden_set.apply(
                    lambda row: self._predict_relation(
                        row,
                        prem_col=self.golden_set_cfg.golden_answer_col,
                        hypo_col=model_cfg.answer_col,
                    ),
                    axis=1,
                )
            )

            # Similarity
            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_embs"] = (
                self._encode_sentences(
                    self.golden_set_cfg.golden_set[model_cfg.answer_col].values
                )
            )

            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_sim"] = (
                self.golden_set_cfg.golden_set.apply(
                    lambda row: self._predict_similarity(
                        row,
                        ref_col=f"{self.golden_set_cfg.golden_answer_col}_embs",
                        col=f"{model_cfg.answer_col}_embs",
                    ),
                    axis=1,
                )
            )

            # Unigram overlap
            overlap_columns = self._get_overlap_metric_columns(
                self.golden_set_cfg.golden_set,
                col_name=model_cfg.answer_col,
                ref_col=self.golden_set_cfg.golden_answer_col,
                col=model_cfg.answer_col,
                ngram="unigram",
            )
            if overlap_columns.columns[0] in self.golden_set_cfg.golden_set.columns:
                self.golden_set_cfg.golden_set = self.golden_set_cfg.golden_set.drop(
                    overlap_columns.columns, axis=1
                )
            self.golden_set_cfg.golden_set = pd.concat(
                [self.golden_set_cfg.golden_set, overlap_columns], axis=1
            )

            # Bigram overlap
            overlap_columns = self._get_overlap_metric_columns(
                self.golden_set_cfg.golden_set,
                col_name=model_cfg.answer_col,
                ref_col=self.golden_set_cfg.golden_answer_col,
                col=model_cfg.answer_col,
                ngram="bigram",
            )
            if overlap_columns.columns[0] in self.golden_set_cfg.golden_set.columns:
                self.golden_set_cfg.golden_set = self.golden_set_cfg.golden_set.drop(
                    overlap_columns.columns, axis=1
                )
            self.golden_set_cfg.golden_set = pd.concat(
                [self.golden_set_cfg.golden_set, overlap_columns], axis=1
            )

            # ROUGE-L
            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_rouge"] = (
                self.golden_set_cfg.golden_set.apply(
                    lambda row: self._calculate_rouge(
                        row,
                        ref_col=self.golden_set_cfg.golden_answer_col,
                        col=model_cfg.answer_col,
                    ),
                    axis=1,
                )
            )

            # BLEU
            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_bleu"] = (
                self.golden_set_cfg.golden_set.apply(
                    lambda row: self._calculate_bleu(
                        row,
                        ref_col=self.golden_set_cfg.golden_answer_col,
                        col=model_cfg.answer_col,
                    ),
                    axis=1,
                )
            )

        total_report = []
        for model_cfg in self.golden_set_cfg.models_cfg:
            rouge_metrics = self.golden_set_cfg.golden_set[
                f"{model_cfg.answer_col}_rouge"
            ].apply(pd.Series)
            report = RAGEReport(
                report_name=model_cfg.answer_col,
                entailment_score=self._nli_score(
                    self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"],
                    mtype="entailment",
                ),
                neutral_score=self._nli_score(
                    self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"],
                    mtype="neutral",
                ),
                contradiction_score=self._nli_score(
                    self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"],
                    mtype="contradiction",
                ),
                similarity_score=self.golden_set_cfg.golden_set[
                    f"{model_cfg.answer_col}_sim"
                ].median(),
                unigram_overlap_precision=self.golden_set_cfg.golden_set[
                    f"{model_cfg.answer_col}_unigram_overlap_precision"
                ].median(),
                unigram_overlap_recall=self.golden_set_cfg.golden_set[
                    f"{model_cfg.answer_col}_unigram_overlap_recall"
                ].median(),
                unigram_overlap_f1=self.golden_set_cfg.golden_set[
                    f"{model_cfg.answer_col}_unigram_overlap_f1"
                ].median(),
                bigram_overlap_precision=self.golden_set_cfg.golden_set[
                    f"{model_cfg.answer_col}_bigram_overlap_precision"
                ].median(),
                bigram_overlap_recall=self.golden_set_cfg.golden_set[
                    f"{model_cfg.answer_col}_bigram_overlap_recall"
                ].median(),
                bigram_overlap_f1=self.golden_set_cfg.golden_set[
                    f"{model_cfg.answer_col}_bigram_overlap_f1"
                ].median(),
                rouge_precision=rouge_metrics[0].median(),
                rouge_recall=rouge_metrics[1].median(),
                rouge_f1=rouge_metrics[2].median(),
                bleu_score=self.golden_set_cfg.golden_set[
                    f"{model_cfg.answer_col}_bleu"
                ].median(),
            )
            total_report.append(report)
            if print_report:
                self.correctness_report(report=report)

        if pointwise_report:
            pointwise_reports = []
            for model_cfg in self.golden_set_cfg.models_cfg:
                metric_columns = [
                    self.golden_set_cfg.question_col,
                    self.golden_set_cfg.golden_answer_col,
                ]
                metric_columns.append(model_cfg.context_col)
                metric_columns.append(model_cfg.answer_col)
                metric_columns.append(f"{model_cfg.answer_col}_nli")
                metric_columns.append(f"{model_cfg.answer_col}_sim")
                metric_columns.append(
                    f"{model_cfg.answer_col}_unigram_overlap_precision"
                )
                metric_columns.append(f"{model_cfg.answer_col}_unigram_overlap_recall")
                metric_columns.append(f"{model_cfg.answer_col}_unigram_overlap_f1")
                metric_columns.append(
                    f"{model_cfg.answer_col}_bigram_overlap_precision"
                )
                metric_columns.append(f"{model_cfg.answer_col}_bigram_overlap_recall")
                metric_columns.append(f"{model_cfg.answer_col}_bigram_overlap_f1")
                metric_columns.append(f"{model_cfg.answer_col}_rouge")
                metric_columns.append(f"{model_cfg.answer_col}_bleu")

                pointwise_reports.append(self.golden_set_cfg.golden_set[metric_columns])

            return total_report, pointwise_reports

        return total_report

    def evaluate_faithfulness(
        self,
        nli_model_name: str = "MTS-AI-SearchSkill/DeBERTa-nli-ru",
        print_report: bool = False,
        pointwise_report: bool = False,
    ) -> List[RAGEReport]:
        """Evaluate models on the faithfulness task (A~C).
        It estimates NLI, unigram overlap (P/R/F1) and ROUGE-L (reversed P) scores.

        Args:
            nli_model_name (str, optional): HF model name to use for the NLI score.
            Defaults to "MTS-AI-SearchSkill/DeBERTa-nli-ru".
            print_report (bool, optional):  Whether to print the output to the console.
            Defaults to True.

        Returns:
            List[RAGEReport]: A list of the reports for each model.
        """
        print("Starting faithfulness evaluation")
        # NLI
        self._init_nli_model(model_name=nli_model_name)
        for model_cfg in tqdm(self.golden_set_cfg.models_cfg, desc="Model #"):
            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"] = (
                self.golden_set_cfg.golden_set.apply(
                    lambda row: self._predict_relation(
                        row,
                        prem_col=model_cfg.context_col,
                        hypo_col=model_cfg.answer_col,
                    ),
                    axis=1,
                )
            )

            # Unigram overlap
            overlap_columns = self._get_overlap_metric_columns(
                self.golden_set_cfg.golden_set,
                col_name=model_cfg.answer_col,
                ref_col=model_cfg.answer_col,
                col=model_cfg.context_col,
                ngram="unigram",
            )
            if overlap_columns.columns[0] in self.golden_set_cfg.golden_set.columns:
                self.golden_set_cfg.golden_set = self.golden_set_cfg.golden_set.drop(
                    overlap_columns.columns, axis=1
                )
            self.golden_set_cfg.golden_set = pd.concat(
                [self.golden_set_cfg.golden_set, overlap_columns], axis=1
            )

            # ROUGE-L
            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_rouge"] = (
                self.golden_set_cfg.golden_set.apply(
                    lambda row: self._calculate_rouge(
                        row,
                        ref_col=model_cfg.context_col,
                        col=model_cfg.answer_col,
                    ),
                    axis=1,
                )
            )

        total_report = []
        for model_cfg in self.golden_set_cfg.models_cfg:
            rouge_metrics = self.golden_set_cfg.golden_set[
                f"{model_cfg.answer_col}_rouge"
            ].apply(pd.Series)
            report = RAGEReport(
                report_name=model_cfg.answer_col,
                entailment_score=self._nli_score(
                    self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"],
                    mtype="entailment",
                ),
                neutral_score=self._nli_score(
                    self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"],
                    mtype="neutral",
                ),
                contradiction_score=self._nli_score(
                    self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"],
                    mtype="contradiction",
                ),
                unigram_overlap_precision=self.golden_set_cfg.golden_set[
                    f"{model_cfg.answer_col}_unigram_overlap_precision"
                ].median(),
                rouge_precision=rouge_metrics[0].median(),
            )
            total_report.append(report)
            if print_report:
                self.faithfulness_report(report=report)

        if pointwise_report:
            pointwise_reports = []
            for model_cfg in self.golden_set_cfg.models_cfg:
                metric_columns = [
                    self.golden_set_cfg.question_col,
                    self.golden_set_cfg.golden_answer_col,
                ]
                metric_columns.append(model_cfg.context_col)
                metric_columns.append(model_cfg.answer_col)
                metric_columns.append(f"{model_cfg.answer_col}_nli")
                metric_columns.append(
                    f"{model_cfg.answer_col}_unigram_overlap_precision"
                )
                metric_columns.append(f"{model_cfg.answer_col}_rouge")

                pointwise_reports.append(self.golden_set_cfg.golden_set[metric_columns])

            return total_report, pointwise_reports

        return total_report

    def evaluate_relevance(
        self,
        sim_model_name: str = "intfloat/multilingual-e5-large",
        print_report: bool = False,
        pointwise_report: bool = False,
    ) -> List[RAGEReport]:
        """Evaluate models on the relevance task (A~Q).
        It estimates similarity, unigram overlap (P) and ROUGE-L (R) scores.

        Args:
            sim_model_name (str, optional): HF model name to use for the Similarity score.
            Defaults to "intfloat/multilingual-e5-large".
            print_report (bool, optional):  Whether to print the output to the console.
            Defaults to True.

        Returns:
            List[RAGEReport]: A list of the reports each model.
        """
        print("Starting relevance evaluation")
        # Similarity
        self._init_sim_model(model_name=sim_model_name)
        self.golden_set_cfg.golden_set[f"{self.golden_set_cfg.question_col}_embs"] = (
            self._encode_sentences(
                self.golden_set_cfg.golden_set[self.golden_set_cfg.question_col].values
            )
        )
        for model_cfg in tqdm(self.golden_set_cfg.models_cfg, desc="Model #"):
            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_embs"] = (
                self._encode_sentences(
                    self.golden_set_cfg.golden_set[model_cfg.answer_col].values,
                    prefix="passage",
                )
            )

            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_sim"] = (
                self.golden_set_cfg.golden_set.apply(
                    lambda row: self._predict_similarity(
                        row,
                        ref_col=f"{self.golden_set_cfg.question_col}_embs",
                        col=f"{model_cfg.answer_col}_embs",
                    ),
                    axis=1,
                )
            )

            # Unigram overlap
            overlap_columns = self._get_overlap_metric_columns(
                self.golden_set_cfg.golden_set,
                col_name=model_cfg.answer_col,
                ref_col=self.golden_set_cfg.question_col,
                col=model_cfg.answer_col,
                ngram="unigram",
            )
            if overlap_columns.columns[0] in self.golden_set_cfg.golden_set.columns:
                self.golden_set_cfg.golden_set = self.golden_set_cfg.golden_set.drop(
                    overlap_columns.columns, axis=1
                )
            self.golden_set_cfg.golden_set = pd.concat(
                [self.golden_set_cfg.golden_set, overlap_columns], axis=1
            )

            # ROUGE-L
            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_rouge"] = (
                self.golden_set_cfg.golden_set.apply(
                    lambda row: self._calculate_rouge(
                        row,
                        ref_col=self.golden_set_cfg.question_col,
                        col=model_cfg.answer_col,
                    ),
                    axis=1,
                )
            )

        total_report = []
        for model_cfg in self.golden_set_cfg.models_cfg:
            rouge_metrics = self.golden_set_cfg.golden_set[
                f"{model_cfg.answer_col}_rouge"
            ].apply(pd.Series)
            report = RAGEReport(
                report_name=model_cfg.answer_col,
                similarity_score=self.golden_set_cfg.golden_set[
                    f"{model_cfg.answer_col}_sim"
                ].median(),
                unigram_overlap_precision=self.golden_set_cfg.golden_set[
                    f"{model_cfg.answer_col}_unigram_overlap_precision"
                ].median(),
                rouge_recall=rouge_metrics[1].median(),
            )
            total_report.append(report)
            if print_report:
                self.relevance_report(report=report)

        if pointwise_report:
            pointwise_reports = []
            for model_cfg in self.golden_set_cfg.models_cfg:
                metric_columns = [
                    self.golden_set_cfg.question_col,
                    self.golden_set_cfg.golden_answer_col,
                ]
                metric_columns.append(model_cfg.context_col)
                metric_columns.append(model_cfg.answer_col)
                metric_columns.append(f"{model_cfg.answer_col}_sim")
                metric_columns.append(
                    f"{model_cfg.answer_col}_unigram_overlap_precision"
                )
                metric_columns.append(f"{model_cfg.answer_col}_rouge")

                pointwise_reports.append(self.golden_set_cfg.golden_set[metric_columns])

            return total_report, pointwise_reports

        return total_report

    def comprehensive_evaluation(
        self,
        nli_model_name: str = "MTS-AI-SearchSkill/DeBERTa-nli-ru",
        sim_model_name: str = "intfloat/multilingual-e5-large",
        print_report: bool = False,
        pointwise_report: bool = False,
    ) -> Tuple[List[RAGEReport]]:
        """Evaluate models on the correctness (A~A*), faithfulness (A~C) and relevance (A~Q) tasks.
        Correctness:  NLI, similarity, uni-/bi-gram overlap (P/R/F1), ROUGE-L (P/R/F1) and BLEU scores.
        Faithfulness: NLI, unigram overlap (P/R/F1) and ROUGE-L (reversed P) scores.
        Relevance: similarity, unigram overlap (P) and ROUGE-L (P) scores.

        Args:
            nli_model_name (str, optional): HF model name to use for the NLI score.
            Defaults to "MTS-AI-SearchSkill/DeBERTa-nli-ru".
            sim_model_name (str, optional): HF model name to use for the Similarity score.
            Defaults to "intfloat/multilingual-e5-large".
            print_report (bool, optional):  Whether to print the output to the console.
            Defaults to True.

        Returns:
            Tuple[List[RAGEReport]]: A list of the reports for each model.
        """
        if pointwise_report:
            correctness_report, correctness_pointwise = self.evaluate_correctness(
                nli_model_name=nli_model_name,
                sim_model_name=sim_model_name,
                print_report=print_report,
                pointwise_report=pointwise_report,
            )
            faithfulness_report, faithfulness_pointwise = self.evaluate_faithfulness(
                nli_model_name=nli_model_name,
                print_report=print_report,
                pointwise_report=pointwise_report,
            )
            relevance_report, relevance_pointwise = self.evaluate_relevance(
                sim_model_name=sim_model_name,
                print_report=print_report,
                pointwise_report=pointwise_report,
            )

            return (
                correctness_report,
                faithfulness_report,
                relevance_report,
                correctness_pointwise,
                faithfulness_pointwise,
                relevance_pointwise,
            )

        correctness_report = self.evaluate_correctness(
            nli_model_name=nli_model_name,
            sim_model_name=sim_model_name,
            print_report=print_report,
            pointwise_report=pointwise_report,
        )
        faithfulness_report = self.evaluate_faithfulness(
            nli_model_name=nli_model_name,
            print_report=print_report,
            pointwise_report=pointwise_report,
        )
        relevance_report = self.evaluate_relevance(
            sim_model_name=sim_model_name,
            print_report=print_report,
            pointwise_report=pointwise_report,
        )

        return correctness_report, faithfulness_report, relevance_report
    
if __name__ == "__main__":
    evaluator = RAGEvaluator()

    # Example usage
    correctness_report = evaluator.evaluate_correctness()
    faithfulness_report = evaluator.evaluate_faithfulness()
    relevance_report = evaluator.evaluate_relevance()

    print("Correctness Report:")
    for report in correctness_report:
        print(report)

    print("Faithfulness Report:")
    for report in faithfulness_report:
        print(report)

    print("Relevance Report:")
    for report in relevance_report:
        print(report)