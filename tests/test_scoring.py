import unittest
from typing import cast

import pandas as pd

from src.scoring import assign_risk_band, probability_to_score


class ScoringTest(unittest.TestCase):
    def test_probability_to_score_boundaries(self):
        self.assertEqual(probability_to_score(0), 850)
        self.assertEqual(probability_to_score(1), 300)
        self.assertEqual(probability_to_score(0.5), 575)
    def test_probability_to_score_with_series(self):
        probabilities = pd.Series([0.0, 0.5, 1.0])

        scores = probability_to_score(probabilities)
        self.assertIsInstance(scores, pd.Series)
        scores = cast(pd.Series, scores)

        self.assertEqual(list(scores), [850.0, 575.0, 300.0])
        self.assertEqual(list(scores), [850.0, 575.0, 300.0])

    def test_assign_risk_band(self):
        self.assertEqual(assign_risk_band(750), "low")
        self.assertEqual(assign_risk_band(850), "low")
        self.assertEqual(assign_risk_band(650), "medium")
        self.assertEqual(assign_risk_band(749), "medium")
        self.assertEqual(assign_risk_band(649), "high")
        self.assertEqual(assign_risk_band(300), "high")


if __name__ == "__main__":
    unittest.main()
