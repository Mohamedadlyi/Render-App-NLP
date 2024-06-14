import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X)
        return X.apply(self.clean_text)

    @staticmethod
    def clean_text(text):
        text = TextCleaner.remove_urls(text)
        text = TextCleaner.remove_mentions(text)
        text = TextCleaner.remove_english_words(text)
        text = TextCleaner.remove_unicode_bmp(text)
        text = TextCleaner.remove_emoji_shortcodes(text)
        text = TextCleaner.remove_specific_punctuation(text)
        text = TextCleaner.remove_complex_patterns(text)
        text = TextCleaner.remove_various_punctuation(text)
        text = TextCleaner.remove_numbers(text)
        text = TextCleaner.remove_extra_spaces(text)
        return text

    @staticmethod
    def remove_urls(text):
        return re.sub(r'http[s]?://\S+', ' ', text)

    @staticmethod
    def remove_mentions(text):
        return re.sub(r'@\w+', ' ', text)

    @staticmethod
    def remove_english_words(text):
        return re.sub(r'\b[a-zA-Z]+\b', ' ', text)

    @staticmethod
    def remove_unicode_bmp(text):
        return re.sub(r'[\U00010000-\U0010ffff]', ' ', text)

    @staticmethod
    def remove_emoji_shortcodes(text):
        return re.sub(r':[a-z_]+:', ' ', text)

    @staticmethod
    def remove_specific_punctuation(text):
        return re.sub(r'[*!?#@]', ' ', text)

    @staticmethod
    def remove_complex_patterns(text):
        return re.sub(r'\|\|+\\s*\d+%\s*\|\|+?[_\-\.\?]+', ' ', text)

    @staticmethod
    def remove_various_punctuation(text):
        return re.sub(r'[_\-\.\"\:\;\,\'\،\♡\\\)/(\&\؟]', ' ', text)

    @staticmethod
    def remove_numbers(text):
        return re.sub(r'\d+', ' ', text)

    @staticmethod
    def remove_extra_spaces(text):
        return ' '.join(text.split())
