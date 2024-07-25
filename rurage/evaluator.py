from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import rouge_score
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .config import RAGEModelConfig, RAGESetConfig
from .metrics import (
    calculate_bleu,
    calculate_rouge,
    compute_nli_score,
    compute_overlap,
    compute_similarity,
)
from .report import (
    RAGEReport,
    correctness_report,
    faithfulness_report,
    relevance_report,
)
from .tokenizer import Tokenizer


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

    def _init_nli_model(self, model_name: str) -> None:
        if model_name != self.nli_model_name:
            print(f"Initializing the NLI model: {model_name}")
            self.nli_model_name = model_name
            self._nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(
                self.nli_model_name
            ).to(self.device)
            self._nli_labels = self._nli_model.config.id2label
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
        else:
            print("The similarity model has alredy been loaded.")

    def _predict_relation(
        self, row: pd.Series, premise_column_name: str, hypothesis_column_name: str
    ) -> float:
        premise = row[premise_column_name]
        hypothesis = row[hypothesis_column_name]
        input = self._nli_tokenizer(
            premise, hypothesis, truncation=True, return_tensors="pt"
        )
        output = self._nli_model(input["input_ids"].to(self.device))
        label = int(torch.argmax(output["logits"]))

        return label

    def _nli_scores(
        self,
        model_cfg: RAGEModelConfig,
        premise_column_name: str,
        hypothesis_column_name: str,
    ) -> Tuple[float]:
        self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"] = (
            self.golden_set_cfg.golden_set.apply(
                lambda row: self._predict_relation(
                    row,
                    premise_column_name=premise_column_name,
                    hypothesis_column_name=hypothesis_column_name,
                ),
                axis=1,
            )
        )
        entailment_score = compute_nli_score(
            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"],
            mtype="entailment",
            class_labels=self._nli_labels,
            norm_size=self.golden_set_cfg.golden_set.shape[0],
        )
        neutral_score = compute_nli_score(
            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"],
            mtype="neutral",
            class_labels=self._nli_labels,
            norm_size=self.golden_set_cfg.golden_set.shape[0],
        )
        contradiction_score = compute_nli_score(
            self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_nli"],
            mtype="contradiction",
            class_labels=self._nli_labels,
            norm_size=self.golden_set_cfg.golden_set.shape[0],
        )

        return entailment_score, neutral_score, contradiction_score

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
        sentence_embeddings = []
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

                sentence_embeddings.extend(embeddings)

        return sentence_embeddings

    @staticmethod
    def _predict_similarity(
        row: pd.Series, reference_column_name: str, column_name: str
    ) -> float:
        return compute_similarity(row[reference_column_name], row[column_name])

    def _similarity_scores(
        self, model_cfg: RAGEModelConfig, reference_column_name: str, column_name: str
    ) -> float:
        self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_embs"] = (
            self._encode_sentences(
                self.golden_set_cfg.golden_set[model_cfg.answer_col].values
            )
        )

        self.golden_set_cfg.golden_set.apply(
            lambda row: self._predict_similarity(
                row,
                reference_column_name=reference_column_name,
                column_name=column_name,
            ),
            axis=1,
        )

        return self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_sim"].median()

    def _compute_overlap(
        self, row: pd.Series, reference_column_name: str, column_name: str, ngram: str
    ) -> Tuple[float]:
        if ngram == "unigram":
            tokenizer = self._unigram_tokenizer
        else:
            tokenizer = self._bigram_tokenizer

        return compute_overlap(
            row[reference_column_name], row[column_name], tokenizer=tokenizer
        )

    def _overlap_score(
        self,
        model_cfg: RAGEModelConfig,
        reference_column_name: str,
        column_name: str,
        ngram: str,
    ) -> Tuple[float]:
        overlap_metrics = self.golden_set_cfg.golden_set.apply(
            lambda row: self._compute_overlap(
                row,
                reference_column_name=reference_column_name,
                column_name=column_name,
                ngram=ngram,
            ),
            axis=1,
        )
        overlap_metrics = pd.DataFrame(
            overlap_metrics.to_list(),
            columns=[
                f"{model_cfg.answer_col}_{ngram}_overlap_precision",
                f"{model_cfg.answer_col}_{ngram}_overlap_recall",
                f"{model_cfg.answer_col}_{ngram}_overlap_f1",
            ],
        )

        if overlap_metrics.columns[0] in self.golden_set_cfg.golden_set.columns:
            self.golden_set_cfg.golden_set = self.golden_set_cfg.golden_set.drop(
                overlap_metrics.columns, axis=1
            )
        self.golden_set_cfg.golden_set = pd.concat(
            [self.golden_set_cfg.golden_set, overlap_metrics], axis=1
        )

        return (
            self.golden_set_cfg.golden_set[
                f"{model_cfg.answer_col}_{ngram}_overlap_precision"
            ].median(),
            self.golden_set_cfg.golden_set[
                f"{model_cfg.answer_col}_{ngram}_overlap_recall"
            ].median(),
            self.golden_set_cfg.golden_set[
                f"{model_cfg.answer_col}_{ngram}_overlap_f1"
            ].median(),
        )

    def _calculate_rouge(
        self, row: pd.Series, reference_column_name: str, column_name: str
    ) -> "rouge_score.scoring.Score":
        reference = row[reference_column_name]
        candidate = row[column_name]
        return calculate_rouge(reference, candidate, rouge_scorer=self._rouge_scorer)

    def _rouge_score(
        self,
        model_cfg: RAGEModelConfig,
        reference_column_name: str,
        column_name: str,
    ) -> Tuple[float]:
        self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_rouge"] = (
            self.golden_set_cfg.golden_set.apply(
                lambda row: self._calculate_rouge(
                    row,
                    reference_column_name=reference_column_name,
                    column_name=column_name,
                ),
                axis=1,
            )
        )
        rouge_metrics = self.golden_set_cfg.golden_set[
            f"{model_cfg.answer_col}_rouge"
        ].apply(pd.Series)
        return (
            rouge_metrics[0].median(),
            rouge_metrics[1].median(),
            rouge_metrics[2].median(),
        )

    def _calculate_bleu(
        self, row: pd.Series, reference_column_name: str, column_name: str
    ) -> float:
        reference = self._unigram_tokenizer(row[reference_column_name])
        candidate = self._unigram_tokenizer(row[column_name])

        return calculate_bleu(reference, candidate)

    def _bleu_score(
        self,
        model_cfg: RAGEModelConfig,
        reference_column_name: str,
        column_name: str,
    ) -> float:
        self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_bleu"] = (
            self.golden_set_cfg.golden_set.apply(
                lambda row: self._calculate_bleu(
                    row,
                    reference_column_name=reference_column_name,
                    column_name=column_name,
                ),
                axis=1,
            )
        )

        return self.golden_set_cfg.golden_set[f"{model_cfg.answer_col}_bleu"].median()

    def evaluate_correctness(
        self,
        nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        sim_model_name: str = "intfloat/multilingual-e5-large",
        print_report: bool = False,
        pointwise_report: bool = False,
    ) -> List[RAGEReport]:
        """Evaluate models on the correctness task (A~A*).
        It estimates NLI, similarity, uni-/bi-gram overlap (P/R/F1), ROUGE-L (P/R/F1) and BLEU scores.

        Args:
            nli_model_name (str, optional): HF model name to use for the NLI score.
            Defaults to "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7".
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
            entailment_score, neutral_score, contradiction_score = self._nli_scores(
                model_cfg,
                premise_column_name=self.golden_set_cfg.golden_answer_col,
                hypothesis_column_name=model_cfg.answer_col,
            )
            # Similarity
            similarity_score = self._predict_similarity(
                model_cfg,
                reference_column_name=f"{self.golden_set_cfg.golden_answer_col}_embs",
                column_name=f"{model_cfg.answer_col}_embs",
            )
            # Unigram overlap
            unigram_overlap_precision, unigram_overlap_recall, unigram_overlap_f1 = (
                self._overlap_score(
                    model_cfg,
                    reference_column_name=self.golden_set_cfg.golden_answer_col,
                    column_name=model_cfg.answer_col,
                    ngram="unigram",
                )
            )
            # Bigram overlap
            bigram_overlap_precision, bigram_overlap_recall, bigram_overlap_f1 = (
                self._overlap_score(
                    model_cfg,
                    reference_column_name=self.golden_set_cfg.golden_answer_col,
                    column_name=model_cfg.answer_col,
                    ngram="bigram",
                )
            )
            # ROUGE-L
            rouge_precision, rouge_recall, rouge_f1 = self._rouge_score(
                model_cfg,
                reference_column_name=self.golden_set_cfg.golden_answer_col,
                column_name=model_cfg.answer_col,
            )
            # BLEU
            bleu_score = self._bleu_score(
                model_cfg,
                reference_column_name=self.golden_set_cfg.golden_answer_col,
                column_name=model_cfg.answer_col,
            )

        total_report = []
        for model_cfg in self.golden_set_cfg.models_cfg:
            report = RAGEReport(
                report_name=model_cfg.answer_col,
                entailment_score=entailment_score,
                neutral_score=neutral_score,
                contradiction_score=contradiction_score,
                similarity_score=similarity_score,
                unigram_overlap_precision=unigram_overlap_precision,
                unigram_overlap_recall=unigram_overlap_recall,
                unigram_overlap_f1=unigram_overlap_f1,
                bigram_overlap_precision=bigram_overlap_precision,
                bigram_overlap_recall=bigram_overlap_recall,
                bigram_overlap_f1=bigram_overlap_f1,
                rouge_precision=rouge_precision,
                rouge_recall=rouge_recall,
                rouge_f1=rouge_f1,
                bleu_score=bleu_score,
            )
            total_report.append(report)
            if print_report:
                correctness_report(report)

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
        nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        print_report: bool = False,
        pointwise_report: bool = False,
    ) -> List[RAGEReport]:
        """Evaluate models on the faithfulness task (A~C).
        It estimates NLI, unigram overlap (P/R/F1) and ROUGE-L (reversed P) scores.

        Args:
            nli_model_name (str, optional): HF model name to use for the NLI score.
            Defaults to "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7".
            print_report (bool, optional):  Whether to print the output to the console.
            Defaults to True.

        Returns:
            List[RAGEReport]: A list of the reports for each model.
        """
        print("Starting faithfulness evaluation")
        # init NLI
        self._init_nli_model(model_name=nli_model_name)
        for model_cfg in tqdm(self.golden_set_cfg.models_cfg, desc="Model #"):
            # NLI
            entailment_score, neutral_score, contradiction_score = self._nli_scores(
                model_cfg,
                premise_column_name=model_cfg.context_col,
                hypothesis_column_name=model_cfg.answer_col,
            )
            # Unigram overlap
            unigram_overlap_precision, _, _ = self._overlap_score(
                model_cfg,
                reference_column_name=model_cfg.answer_col,
                column_name=model_cfg.context_col,
                ngram="unigram",
            )
            # ROUGE-L
            rouge_precision, _, _ = self._rouge_score(
                model_cfg,
                reference_column_name=model_cfg.context_col,
                column_name=model_cfg.answer_col,
            )

        total_report = []
        for model_cfg in self.golden_set_cfg.models_cfg:
            report = RAGEReport(
                report_name=model_cfg.answer_col,
                entailment_score=entailment_score,
                neutral_score=neutral_score,
                contradiction_score=contradiction_score,
                unigram_overlap_precision=unigram_overlap_precision,
                rouge_precision=rouge_precision,
            )
            total_report.append(report)
            if print_report:
                faithfulness_report(report)

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
            similarity_score = self._predict_similarity(
                model_cfg,
                reference_column_name=f"{self.golden_set_cfg.question_col}_embs",
                column_name=f"{model_cfg.answer_col}_embs",
            )
            # Unigram overlap
            unigram_overlap_precision, _, _ = self._overlap_score(
                model_cfg,
                reference_column_name=model_cfg.answer_col,
                column_name=self.golden_set_cfg.question_col,
                ngram="unigram",
            )

            # ROUGE-L
            _, rouge_recall, _ = self._rouge_score(
                model_cfg,
                reference_column_name=self.golden_set_cfg.question_col,
                column_name=model_cfg.answer_col,
            )

        total_report = []
        for model_cfg in self.golden_set_cfg.models_cfg:
            report = RAGEReport(
                report_name=model_cfg.answer_col,
                similarity_score=similarity_score,
                unigram_overlap_precision=unigram_overlap_precision,
                rouge_recall=rouge_recall,
            )
            total_report.append(report)
            if print_report:
                relevance_report(report)

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
        nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        sim_model_name: str = "intfloat/multilingual-e5-large",
        print_report: bool = False,
        pointwise_report: bool = False,
    ) -> Dict[str, List[RAGEReport]]:
        """Evaluate models on the correctness (A~A*), faithfulness (A~C) and relevance (A~Q) tasks.
        Correctness:  NLI, similarity, uni-/bi-gram overlap (P/R/F1), ROUGE-L (P/R/F1) and BLEU scores.
        Faithfulness: NLI, unigram overlap (P/R/F1) and ROUGE-L (reversed P) scores.
        Relevance: similarity, unigram overlap (P) and ROUGE-L (P) scores.

        Args:
            nli_model_name (str, optional): HF model name to use for the NLI score.
            Defaults to "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7".
            sim_model_name (str, optional): HF model name to use for the Similarity score.
            Defaults to "intfloat/multilingual-e5-large".
            print_report (bool, optional):  Whether to print the output to the console.
            Defaults to True.

        Returns:
            Dict[str, List[RAGEReport]]: Dict of the reports for the each task and model.
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

            return {
                "correctness_report": correctness_report,
                "faithfulness_report": faithfulness_report,
                "relevance_report": relevance_report,
                "correctness_pointwise": correctness_pointwise,
                "faithfulness_pointwise": faithfulness_pointwise,
                "relevance_pointwise": relevance_pointwise,
            }

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

        return {
            "correctness_report": correctness_report,
            "faithfulness_report": faithfulness_report,
            "relevance_report": relevance_report,
        }
