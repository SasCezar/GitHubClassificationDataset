import csv
from typing import Iterable, List, Dict, Set

import pandas
from pandas import DataFrame
from reconciler import reconcile

from src.processing.preprocessor import AbstractPreprocessing


def normalize(text: str) -> str:
    return str(text).replace("-", "_").lower()


class WikiRedirectNormalizer(AbstractPreprocessing):
    def __init__(self, file_path: str):
        super().__init__()
        self.mapping = self.load_redirects(file_path)

    def filter(self, elements: Iterable) -> List:
        remaining = list()
        for element in elements:
            normalized = normalize(element)
            if normalized in self.mapping:
                standard_form = self.mapping[normalized]
                remaining.append(standard_form)
            else:
                remaining.append(-1)

        return remaining

    @staticmethod
    def load_redirects(path: str) -> Dict:
        mapping = {}
        with open(path, 'rt') as inf:
            reader = csv.reader(inf)
            for src, trg in reader:
                mapping[normalize(src)] = trg

        return mapping


class WikiReconciler(AbstractPreprocessing):
    def __init__(self):
        super().__init__()

    def filter(self, elements: Iterable, top: int = 5) -> DataFrame:
        series = pandas.Series(elements)
        reconciled = reconcile(series, top_res=top)
        return reconciled
