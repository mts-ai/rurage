import unittest

import pandas as pd

from rurage.config import RAGEModelConfig, RAGESetConfig
from rurage.evaluator import RAGEvaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({"question": ["q1"], "answer": ["a1"]})
        model_config = RAGEModelConfig(context_col="context", answer_col="answer")
        set_config = RAGESetConfig(df, "question", "answer", [model_config])
        self.evaluator = RAGEvaluator(set_config)

    def test_evaluate_correctness(self):
        report = self.evaluator.evaluate_correctness()
        self.assertIsNotNone(report)

    def test_evaluate_faithfulness(self):
        report = self.evaluator.evaluate_faithfulness()
        self.assertIsNotNone(report)

    def test_evaluate_relevance(self):
        report = self.evaluator.evaluate_relevance()
        self.assertIsNotNone(report)


if __name__ == "__main__":
    unittest.main()
