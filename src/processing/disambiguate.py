import pandas

from src.processing.preprocessor import AbstractProcessing


class MatchDisambiguate(AbstractProcessing):
    @property
    def name(self):
        return 'match'

    def run(self, reconciled):
        disambiguated = []

        for rec in reconciled:
            rec['best'] = {}
            for candidate in rec['candidates']:
                if candidate['match']:
                    best_match = candidate
                    rec['best'] = best_match
                    break

            disambiguated.append(rec)
        return disambiguated


class FirstDisambiguate(AbstractProcessing):
    @property
    def name(self):
        return 'first'

    def run(self, reconciled):
        disambiguated = []

        for rec in reconciled:
            rec['best'] = rec['candidates'][0] if rec['candidates'] else {}
            disambiguated.append(rec)

        return disambiguated


class FirstCleanDisambiguate(AbstractProcessing):
    def __init__(self, types_path='annotated_wikitopics.csv'):
        super().__init__()
        self.skip_types = self.load_skip_types(types_path)

    @property
    def name(self):
        return 'first-clean'

    def run(self, reconciled):
        disambiguated = []

        for rec in reconciled:
            candidates = [x for x in rec['candidates'] if self.is_clean(x)]
            rec['best'] = candidates[0] if candidates else {}
            disambiguated.append(rec)

        return disambiguated

    def is_clean(self, x):
        types = {k for k in x['types']}
        clean = bool(types.intersection(self.skip_types))
        return not clean

    @staticmethod
    def load_skip_types(path):
        df = pandas.read_csv(path)
        skip = {x for x, v in zip(df['id'], df['skip']) if v == 0}
        return skip
