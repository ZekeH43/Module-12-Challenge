import unittest

import pandas as pd

from credit_risk import TARGET, balance_training, score


class ComparisonTests(unittest.TestCase):
    def test_oversampling_preserves_rows_and_only_duplicates_minority(self):
        train = pd.DataFrame({TARGET: [0, 0, 0, 1], "income": [10, 20, 30, 40]})
        result = balance_training(train)
        self.assertEqual(result[TARGET].value_counts().to_dict(), {0: 3, 1: 3})
        self.assertEqual(result.index.tolist(), [0, 1, 2, 3, 3, 3])
        self.assertEqual(len(train), 4)

    def test_error_counts_have_the_right_direction(self):
        metrics = score([0, 0, 0, 1, 1], [1, 1, 0, 0, 1])
        self.assertEqual(metrics["false positives"], 2)
        self.assertEqual(metrics["missed high-risk"], 1)
        self.assertAlmostEqual(metrics["precision (1)"], 1 / 3)
        self.assertEqual(metrics["recall (1)"], 0.5)
