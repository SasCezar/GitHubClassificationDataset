from abc import ABC
from typing import Iterable, List


class AbstractProcessing(ABC):
    def __init__(self):
        pass

    def run(self, elements: Iterable) -> List:
        pass
