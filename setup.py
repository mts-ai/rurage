from setuptools import setup, find_packages

setup(
    name="RuRAGE",
    version="1.0.0",
    description="RuRAGE (Russian RAG Evaluation) is a Python library developed to speed-up evaluation russian RAG systems with Correctness, Faithfulness and Relevance axes using a variety of deterministic and model-based metrics.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sidorov Ivan MTS AI, Kraiko Nikita MTS AI",
    author_email="i.sidorov1@mts.ai, n.kraiko@mts.ai",
    url="https://github.com/mts-ai/rurage",
    packages=find_packages(where="rurage"),
    package_dir={"": "rurage"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "nltk",
        "pymorphy2",
        "rouge_score",
        "transformers",
        "torch",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
