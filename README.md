# RuRAGE - Russian RAG Evaluation

RuRAGE (Russian RAG Evaluation) is a Python library developed to speed-up evaluation russian RAG systems with Correctness, Faithfulness and Relevance axes using a variety of deterministic and model-based metrics.

## Features

- **Deterministic Metrics**:
  - ROUGE
  - BLEU
  - Bigram overlap Precision
  - Bigram overlap Recall
  - Bigram overlap F1
  - Unigram overlap Precision
  - Unigram overlap Recall
  - Unigram overlap F1

- **Model-based Metrics**:
  - NLI Scores using Transformer models
  - Cosine Similarity using Transformer models
  - Uncertainty (soon)

- **Ensemble Creation**:
  - Combine scores from multiple metrics to create a robust evaluation ensemble.

## Installation

You can install RuRAGE from PyPI:

```bash
pip install rurage
```

## Usage

Basic example of how to use RuRAGE:

```python
from rurage import Tokenizer, RAGEModelConfig, RAGESetConfig, RAGEReport

tokenizer = Tokenizer(
    lower_case=True,
    remove_punctuation=True,
    remove_stopwords=True,
    lemm=True,
    stem=False,
    language="russian",
    ngram_range=(1, 1)
)

tokens = tokenizer("Ваш текст для токенизации здесь")

model_config = RAGEModelConfig(context_col="context", answer_col="answer")

rage_set_config = RAGESetConfig(
    golden_set=your_dataframe,
    question_col="question",
    golden_answer_col="golden_answer",
    models_cfg=[model_config]
)

report = RAGEReport(report_name="Evaluation Report")

print(tokens)
```

## To-Do List

### By the End of Q3

- **Automatic Ensemble Creation**: Implement functionality for automatic creation of evaluation ensembles.
- **Multiclass Labels**: Extend support to work with multiclass usefulness labels.
- **Uncertainty scores**: Uncertainty scores to ensemble.

### By the End of the Year

- **Judge LLM**: Introduce our proprietary Judge LLM model for enhanced evaluation.

## Contributing

We welcome contributions from the community. Please read our contributing guidelines and code of conduct.

## License

RuRAGE is licensed under the MIT License. See the LICENSE file for more information.

## Contact

For any questions, issues, or suggestions, please open an issue on our [GitHub repository](https://github.com/mts-ai/rurage).

## Acknowledgments

RuRAGE presented in PyCon 2024 by MTS AI Search Group.

Developed by MTS AI Search Group (Krayko Nikita, Laputin Fedor, Sidorov Ivan)