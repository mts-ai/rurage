from setuptools import setup, find_packages

setup(
    name="RuRAGE",
    version="1.0.0",
    description="A library to provide RAG model evaluation reports",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Krayko Nikita MTS AI",
    author_email="n.kraiko@mts.ai",
    url="https://github.com/mts-ai/rurage",
    packages=find_packages(),
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
