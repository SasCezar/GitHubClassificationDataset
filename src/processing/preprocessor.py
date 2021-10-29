from abc import ABC
from typing import Iterable, List


class AbstractPreprocessing(ABC):
    def __init__(self):
        pass

    def filter(self, elements: Iterable) -> List:
        pass
