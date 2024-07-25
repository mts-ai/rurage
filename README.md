# RuRAGE - Russian RAG Evaluation

RuRAGE (Russian RAG Evaluation) is a Python library developed to speed-up evaluation russian RAG systems with Correctness, Faithfulness and Relevance axes using a variety of deterministic and model-based metrics.

Keypoints:

- If there are many weak metrics, you can combine them into an ensemble
- We train ensemble on the necessary nature of the data, carefully without the data leaks of validation
- It is necessary to prepare a Golden-set with standard answers. It is needed for both Judge LLM and RuRAGE
- The resulting usefulness of deterministic metrics almost doubles

Metrics by Mistral 7B top-10 of both approaches evaluated on the Golden set and compared using the best thresholds in classification with usefulness labels by human evaluation. Each metric has its own marker, with classes grouped by color. The strongest metrics are located at the top right according to the axes Recall and Precision.
![Alt text](/docs/image_classification.png)

Metrics of both approaches evaluated on the golden set and compared using Pearson’s correlation with human evaluation (Usefulness) labels. Top-5 and top-10 indicate the number of search engine snippets passed to the model as context. Variations with different refuse rates (No info) from Mistral 7B are included.
![Alt text](/docs/image_bars.png)

Unfortunately, ensemble creation hasn't been added yet, but you can independently experiment with different boostings on decision trees yourself.

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
- **Auto-adaptive thresholds**: Implement functionality for automatic creation thresholds for features in ensemble.
- **Multiclass Labels**: Extend support to work with multiclass usefulness labels.

### By the End of the Year

- **Uncertainty scores**: Uncertainty scores to ensemble.
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