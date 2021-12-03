from abc import ABC, abstractmethod

import numpy
import spacy
import fasttext as ft
from gensim.models import KeyedVectors


class AbstractEmbeddingModel(ABC):
    def __init__(self):
        self._name = 'AbstractEmbeddingModel'
        self.model = None

    @property
    def name(self):
        return self._name

    @abstractmethod
    def get_embedding(self, text: str) -> numpy.ndarray:
        pass


class BERTEmbedding(AbstractEmbeddingModel):
    def __init__(self, model):
        super().__init__()
        self._name = f'{model}'
        self.model = spacy.load(model, disable=["ner", "textcat", "parser"])

    def get_embedding(self, text: str) -> numpy.ndarray:
        return self.model(text).vector


class FastTextEmbedding(AbstractEmbeddingModel):
    def __init__(self, path, model='fastText'):
        super().__init__()
        self._name = f'{model}'
        self.model = ft.load_model(path)

    def get_embedding(self, text: str) -> numpy.ndarray:
        return self.model.get_sentence_vector(text)


class W2VEmbedding(AbstractEmbeddingModel):
    def __init__(self, path, size=100, model='W2V-Unk'):
        super().__init__()
        self._name = f'{model}'
        self.model = KeyedVectors.load_word2vec_format(path)
        self.size = size

    def get_embedding(self, text: str) -> numpy.ndarray:
        embeddings = [self.model.get_vector(x.upper()) for x in text.split(' ') if x in self.model]
        if not embeddings:
            res = numpy.random.random(self.size)
        else:
            res = numpy.mean(embeddings, axis=0)

        return res
