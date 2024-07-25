import re
import string
import warnings
from typing import List, Literal, Tuple

import nltk
import pymorphy2
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

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
