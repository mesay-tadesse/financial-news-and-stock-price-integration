from typing import List, Dict, Any, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora, models
from pandarallel import pandarallel
from numba import jit
import swifter
import dask.dataframe as dd
from tqdm import tqdm
import random

pandarallel.initialize(progress_bar=True)

class TextAnalyzer:
    def __init__(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
            self.stop_words = set()

    @staticmethod
    def _calculate_stats(lengths: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        if isinstance(lengths, pd.Series):
            lengths = lengths.to_numpy()
        if lengths.size == 0:
            return {
                'mean_length': 0.0,
                'max_length': 0.0,
                'min_length': 0.0,
                'total_texts': 0
            }
        return {
            'mean_length': float(np.mean(lengths)),
            'max_length': float(np.max(lengths)),
            'min_length': float(np.min(lengths)),
            'total_texts': int(lengths.size)
        }

    def preprocess_text(self, text: Union[str, pd.Series]) -> Union[List[str], pd.Series]:
        if isinstance(text, pd.Series):
            return text.apply(self._preprocess_single_text)
        return self._preprocess_single_text(text)

    def _preprocess_single_text(self, text: str) -> List[str]:
        tokens = word_tokenize(str(text).lower())
        return [token for token in tokens if token.isalpha() and token not in self.stop_words]

    def get_text_statistics(self, texts: Union[List[str], pd.Series]) -> Dict[str, Any]:
        if isinstance(texts, pd.Series):
            if len(texts) > 1_000_000:
                ddf = dd.from_pandas(pd.DataFrame({'text': texts}), npartitions=4)
                lengths = ddf['text'].str.len().compute()
                print("analyzed huge DS")
            else:
                lengths = texts.str.len().values
                print("analyzed min DS")
        else:
            lengths = np.array([len(str(text)) for text in texts])
        return self._calculate_stats(lengths)

    def extract_common_words(self, texts: Union[List[str], pd.Series], top_n: int = 10, batch_size: int = 10000) -> List[tuple]:
        if isinstance(texts, pd.Series):
            counter = Counter()
            for i in tqdm(range(0, len(texts), batch_size), desc="Processing text batches"):
                batch = texts.iloc[i:i + batch_size]
                tokens = self.preprocess_text(batch)
                batch_tokens = [token for token_list in tokens for token in token_list]
                counter.update(batch_tokens)
        else:
            all_tokens = []
            for text in tqdm(texts, desc="Processing texts"):
                all_tokens.extend(self.preprocess_text(text))
            counter = Counter(all_tokens)
        return counter.most_common(top_n)

    def generate_wordcloud(self, texts: List[str], **kwargs) -> WordCloud:
        combined_text = ' '.join([' '.join(self.preprocess_text(text)) for text in texts])
        return WordCloud(width=800, height=400, background_color='white', **kwargs).generate(combined_text)

    def perform_topic_modeling(self, texts: List[str], num_topics: int = 5, sample_size: int = 10000) -> tuple:
        if len(texts) > sample_size:
            texts = random.sample(list(texts), sample_size)
        processed_texts = [self.preprocess_text(text) for text in tqdm(texts, desc="Preprocessing for LDA")]
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        lda_model = models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            workers=4
        )
        return lda_model, corpus, dictionary
