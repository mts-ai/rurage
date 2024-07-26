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
import pandas as pd

from rurage import RAGEModelConfig, RAGESetConfig, RAGEvaluator

# For each model that needs to be evaluated, you need to initialize a config containing:
# * the name of the column with the context on which the answer was generated
# * the name of the column with the generated model answer
models_cfg = []
models_cfg.append(
    RAGEModelConfig(context_col="example_context_top5", answer_col="model_1_answer")
)
models_cfg.append(
    RAGEModelConfig(context_col="example_context_top5", answer_col="model_2_answer")
)

# Initialize the configuration of the evaluation set:
# * validation set pd.Daraframe
# * The name of the question column
# * The name of the golden answer column
# * The list of model configs
validation_set = pd.read_csv("example_set.csv")
validation_set_cfg = RAGESetConfig(
    golden_set=validation_set,
    question_col="question",
    golden_answer_col="golden_answer",
    models_cfg=models_cfg,
)

# Initialize the evaluator
rager = RAGEvaluator(golden_set_cfg=validation_set_cfg)

# Run a comprehensive evalution (Correctness, Faithfulness, Relevance) for each model
correctness_report, faithfulness_report, relevance_report = (
    rager.comprehensive_evaluation()
)

# Or you can run a separate evaluation
correctness_report = rager.evaluate_correctness()
faithfulness_report = rager.evaluate_faithfulness()
relevance_report = rager.evaluate_relevance()

# For each evaluation method, it is possible to print a report, as well as receive a pointwise report:
# print_report : bool, optional
# Whether to print the output to the console. Defaults to False.

# pointwise_report : bool, optional
# Whether to return pointwise report. Defaults to False.
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