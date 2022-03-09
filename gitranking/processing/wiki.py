import os.path
from typing import Iterable

import pandas
from reconciler import reconcile

from processing.preprocessor import AbstractProcessing


class WikiReconciler(AbstractProcessing):
    def __init__(self, out_path):
        super().__init__()
        self.out = out_path

    def run(self, elements: Iterable, top: int = 5):
        for x in elements:
            try:
                reconciled = reconcile(pandas.Series([x]), top_res=top, property_mapping={"P9100": pandas.Series([x])})
                reconciled.to_csv(os.path.join(self.out, f'reconciled_{top}_{hash(x)}.txt'), index=False)
            except Exception as e:
                print(e)
                print(f'Skipped: {x}')
                continue
