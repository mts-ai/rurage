import unittest

import pandas as pd

from rurage.config import RAGEModelConfig, RAGESetConfig


class TestConfig(unittest.TestCase):
    def test_ragemodelconfig(self):
        config = RAGEModelConfig(context_col="context", answer_col="answer")
        self.assertEqual(config.context_col, "context")

    def test_ragesetconfig(self):
        df = pd.DataFrame({"question": ["q1"], "answer": ["a1"]})
        model_config = RAGEModelConfig(context_col="context", answer_col="answer")
        set_config = RAGESetConfig(df, "question", "answer", [model_config])
        self.assertEqual(set_config.question_col, "question")


if __name__ == "__main__":
    unittest.main()
