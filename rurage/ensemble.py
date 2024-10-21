import os
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError
from sklearn.metrics import f1_score

from .config import RAGESetConfig, ensemble_features


class RAGEnsemble:
    def __init__(
        self,
        ensemble_type: Literal["correctness", "faithfulness", "relevance"],
        model_path: str = None,
    ) -> None:
        self.ensemble_type = ensemble_type
        if model_path:
            try:
                self.model = CatBoostClassifier().load_model(model_path)
            except CatBoostError:
                raise ValueError(
                    f"Failed to load model from {model_path}. Check the path or model format."
                )
        else:
            self.model = CatBoostClassifier(allow_writing_files=False)
        self._threshold = None
        self._class_labels = None
        self._features_dict = ensemble_features
        if self.ensemble_type in set(self._features_dict.keys()):
            self._features = self._features_dict[self.ensemble_type]
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")

    def _split_data(
        self, Xs: List[pd.DataFrame], ys: List[np.array], test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.array, np.array]:
        idxs = Xs[0].sample(frac=1, random_state=42).index
        split_idx = int(Xs[0].shape[0] * (1 - test_size))

        X_train = pd.DataFrame(columns=Xs[0].columns)
        X_test = pd.DataFrame(columns=Xs[0].columns)
        y_train, y_test = np.array([]), np.array([])
        for X, y in zip(Xs, ys):
            X_shuffled = X.iloc[idxs].reset_index(drop=True)
            y_shuffled = y[idxs]

            X_train_ = X_shuffled.iloc[:split_idx]
            X_test_ = X_shuffled.iloc[split_idx:]
            y_train_ = y_shuffled[:split_idx]
            y_test_ = y_shuffled[split_idx:]

            X_train = pd.concat([X_train, X_train_])
            X_test = pd.concat([X_test, X_test_])
            y_train = np.concatenate([y_train, y_train_])
            y_test = np.concatenate([y_test, y_test_])

            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

        return X_train, X_test, y_train, y_test

    def prepare_data_for_study(
        self,
        pointwise_reports: List[pd.DataFrame],
        labels: List[np.array],
        set_config: RAGESetConfig,
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.array, np.array]:
        """Prepare RuRAGE pointwise reports for the ensemble model.

        Args:
            pointwise_reports (List[pd.DataFrame]): pointwise reports from RuRAGE.
            labels (List[np.array]): markup for the used dataset.
            set_config (RAGESetConfig): configuration data for the used dataset.
            test_size (float, optional): The size of the test set. Defaults to 0.2.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, np.array, np.array]: X_train, y_train, X_test, y_test.
        """
        to_drop_columns = [set_config.question_col, set_config.golden_answer_col]
        for model_cfg in set_config.models_cfg:
            to_drop_columns.append(model_cfg.answer_col)
            to_drop_columns.append(model_cfg.context_col)

        Xs, ys = [], []
        for pointwise_report, label in zip(pointwise_reports, labels):
            pointwise_report = pointwise_report.drop(
                to_drop_columns, axis=1, errors="ignore"
            )
            pointwise_report.columns = self._features
            Xs.append(pointwise_report)
            ys.append(label)

        X_train, X_test, y_train, y_test = self._split_data(Xs, ys, test_size=test_size)

        return X_train, y_train, X_test, y_test

    def prepare_data_for_inference(
        self, data: pd.DataFrame, set_config: RAGESetConfig
    ) -> pd.DataFrame:
        """Prepare data for the ensemble model inference.

        Args:
            data (pd.DataFrame): RuRAGE pointwise report (one or more samples).
            set_config (RAGESetConfig): configuration for the used data.

        Returns:
            pd.DataFrame: Prepared data for the ensemble model inference.
        """
        to_drop_columns = [set_config.question_col, set_config.golden_answer_col]
        for model_cfg in set_config.models_cfg:
            to_drop_columns.append(model_cfg.answer_col)
            to_drop_columns.append(model_cfg.context_col)

        X = data.drop(to_drop_columns, axis=1, errors="ignore")
        X.columns = self._features

        return X

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_test: pd.DataFrame = None,
        y_test: np.array = None,
        optimize: bool = True,
    ) -> None:
        """Train the ensemble model on the training data.

        Args:
            X_train (pd.DataFrame): A training set with the RuRAGE features.
            y_train (np.array): A vector with the target values for the train set.
            X_test (pd.DataFrame, optional): A test set with the RuRAGE features. It uses for optimization.
            Defaults to None.
            y_test (np.array, optional): A vector with the target values for the test set.
            It uses for optimization. Defaults to None.
            optimize (bool, optional): Whether to optimize the threshold for the binary/multiclass
            classification task. Defaults to True.
        """
        if set(X_train.columns) != set(self._features):
            raise ValueError(
                f"Dataset for the task '{self.ensemble_type}' must contain columns: '{self._features}'"
            )
        if (X_test is not None) and (y_test is not None):
            eval_set = (X_test, y_test)
            if set(X_test.columns) != set(self._features):
                raise ValueError(
                    f"Dataset for the task '{self.ensemble_type}' must contain columns: '{self._features}'"
                )
        else:
            eval_set = None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            use_best_model=True,
            verbose=False,
        )
        self._class_labels = self.model.classes_

        if optimize:
            self._optimize(X_test, y_test)

    def _optimize(self, X_test: pd.DataFrame, y_test: np.array) -> None:
        print("Starting threshold optimization")
        if len(self._class_labels) == 2:
            y_test = y_test.astype(int)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            thresholds = np.linspace(0, 1, 1000)
            threshold_optim = 0.0
            f1_optim = 0.0
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_test, y_pred)
                if f1 > f1_optim:
                    threshold_optim = threshold
                    f1_optim = f1
            self._threshold = threshold_optim
            print(f"Threshold for the {self.ensemble_type} task: {self._threshold}")
        elif len(self._class_labels) > 2:
            y_pred_probas = self.model.predict(X_test, prediction_type="RawFormulaVal")
            thresholds = np.linspace(-5, 5, 1000)
            classes_thresholds = []
            for i, label in enumerate(self._class_labels):
                y_test_one_vs_all = y_test.copy()
                y_test_one_vs_all[y_test_one_vs_all != label] = 0
                y_test_one_vs_all[y_test_one_vs_all == label] = 1
                y_test_one_vs_all = y_test_one_vs_all.astype(int)
                y_pred_proba = y_pred_probas[:, i]

                threshold_optim = 0.0
                f1_optim = 0.0
                for threshold in thresholds:
                    y_pred = (y_pred_proba >= threshold).astype(int)
                    f1 = f1_score(y_test_one_vs_all, y_pred)
                    if f1 > f1_optim:
                        threshold_optim = threshold
                        f1_optim = f1
                classes_thresholds.append(threshold_optim)
                print(f"Threshold for the '{label}' class: {threshold_optim}")
            self._threshold = classes_thresholds
        else:
            raise ValueError("Number of classes must be 2 or more.")

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """Predict the target values for the test set.

        Args:
            X_test (pd.DataFrame): A test set with the RuRAGE features.

        Returns:
            np.array: A vector with the predicted target values.
        """
        if set(X_test.columns) != set(self._features):
            raise ValueError(
                f"Dataset for the task '{self.ensemble_type}' must contain columns: '{self._features}'"
            )
        if self._threshold is not None:
            if len(self._class_labels) == 2:
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba >= self._threshold).astype(int)
            elif len(self._class_labels) > 2:
                y_pred_probas = self.model.predict(
                    X_test, prediction_type="RawFormulaVal"
                )
                preds_mask = []
                for i in range(len(self._class_labels)):
                    y_pred_proba = y_pred_probas[:, i]
                    y_pred_proba = (y_pred_proba > self._threshold[i]).astype(int)
                    preds_mask.append(y_pred_proba)
                preds_mask = np.stack(preds_mask, axis=1)

                preds_mask_fixed = np.zeros_like(preds_mask)
                y_pred = []
                for i in range(preds_mask.shape[0]):
                    label_row = preds_mask[i]

                    if np.sum(label_row) > 1:
                        max_index = np.argmax(y_pred_probas[i] * label_row)
                        preds_mask_fixed[i, max_index] = 1
                    elif np.sum(label_row) == 0:
                        max_index = np.argmax(y_pred_probas[i])
                        preds_mask_fixed[i, max_index] = 1
                    else:
                        preds_mask_fixed[i] = label_row

                    y_pred.append(self._class_labels[np.argmax(preds_mask_fixed[i])])
            else:
                raise ValueError("Number of classes must be 2 or more.")
        else:
            y_pred = self.model.predict(X_test)

        return y_pred

    def save_model(self, path: str) -> None:
        """Save the model to the specified path.

        Args:
            path (str): A path to the file where the model will be saved.
        """
        extension = os.path.splitext(path)[-1]
        if not extension:
            path += ".cbm"
        try:
            self.model.save_model(path)
        except CatBoostError:
            raise ValueError(
                f"Failed to save model to {path}. Check the path or model format."
            )
