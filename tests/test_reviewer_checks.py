import unittest

import numpy as np

from src.reviewer_checks import near_duplicates, select_threshold, threshold_metrics, word_shingles


class ReviewerChecksTests(unittest.TestCase):
    def test_threshold_selection_uses_validation_and_deterministic_ties(self):
        labels = np.array([0, 0, 1, 1])
        selected = select_threshold(labels, np.array([.1, .2, .3, .4]))
        self.assertEqual(selected['threshold'], .3)
        self.assertEqual(selected['validation_macro_f1'], 1.)
        # Evaluating a separate test array cannot change the selected threshold.
        result = threshold_metrics(labels, np.array([.1, .2, .6, .7]), selected['threshold'])
        self.assertEqual(result['threshold'], .3)
        self.assertEqual(result['confusion_matrix'], [[2, 0], [0, 2]])
        self.assertEqual(select_threshold([0, 1], np.array([.1, .9]))['threshold'], .5)

    def test_near_duplicate_prefix_filter_equals_exhaustive_similarity(self):
        rng = np.random.default_rng(8)
        originals = [' '.join(f'w{x}' for x in rng.integers(0, 30, 60)) for _ in range(25)]
        queries = []
        for text in originals:
            words = text.split()
            for edits in [0, 1, 2, 4]:
                changed = words.copy()
                for i in rng.choice(len(words), edits, replace=False):
                    changed[i] = 'changed'
                queries.append(' '.join(changed))
        queries += ['unrelated short text', '']
        refs = [word_shingles(t) for t in originals]
        for threshold in [.5, .8, 1.]:
            expected = set()
            for i, text in enumerate(queries):
                s = word_shingles(text)
                if any(s and len(s & r) / len(s | r) >= threshold for r in refs):
                    expected.add(i)
            actual, _ = near_duplicates(originals, queries, threshold)
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
