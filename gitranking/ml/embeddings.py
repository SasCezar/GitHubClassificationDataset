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

    def __contains__(self, item):
        return item in self.model


class BERTEmbedding(AbstractEmbeddingModel):
    def __init__(self, model):
        super().__init__()
        self._name = f'{model}'
        self.model = spacy.load(model, disable=["ner", "textcat", "parser"])

    def get_embedding(self, text: str) -> numpy.ndarray:
        return self.model(text).vector

    def __contains__(self, item):
        return True


class FastTextEmbedding(AbstractEmbeddingModel):
    def __init__(self, path, model='fastText'):
        super().__init__()
        self._name = f'{model}'
        self.model = ft.load_model(path)

    def get_embedding(self, text: str) -> numpy.ndarray:
        return self.model.get_sentence_vector(text.lower())

    def __contains__(self, item):
        return True


class W2VEmbedding(AbstractEmbeddingModel):
    def __init__(self, path, size=100, model='W2V-Unk', preprocessing='lower'):
        super().__init__()
        self._name = f'{model}'
        self.model = KeyedVectors.load_word2vec_format(path)
        self.size = size
        self.preprocessing = preprocessing

    def get_embedding(self, text: str) -> numpy.ndarray:
        text = text.lower() if self.preprocessing == 'lower' else text.upper()
        embeddings = [self.model.get_vector(x) for x in text.split(' ') if x in self.model]
        if not embeddings:
            res = numpy.random.random(self.size)
        else:
            res = numpy.mean(embeddings, axis=0)

        return res
