import unittest
from typing import cast

import pandas as pd

from src.scoring import (
    assign_risk_band,
    probability_to_score,
    probability_to_scorecard_points,
)


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

    def test_scorecard_points_pdo_scaling(self):
        # odds == odds_ref (50) -> score_ref (600)
        self.assertEqual(probability_to_scorecard_points(1 / 51), 600)
        # doubling the good/bad odds adds PDO points (+20)
        self.assertEqual(probability_to_scorecard_points(1 / 101), 620)

    def test_scorecard_points_monotonic_and_clip(self):
        high_risk = probability_to_scorecard_points(0.95)
        low_risk = probability_to_scorecard_points(0.02)
        self.assertLess(high_risk, low_risk)  # maior prob -> menor score
        self.assertGreaterEqual(probability_to_scorecard_points(1.0), 300)
        self.assertLessEqual(probability_to_scorecard_points(0.0), 850)

    def test_scorecard_points_with_series(self):
        probabilities = pd.Series([0.02, 0.5, 0.95], index=[10, 20, 30])

        points = probability_to_scorecard_points(probabilities)
        self.assertIsInstance(points, pd.Series)
        self.assertEqual(list(points.index), [10, 20, 30])
        self.assertTrue(points.iloc[0] > points.iloc[1] > points.iloc[2])

    def test_assign_risk_band(self):
        self.assertEqual(assign_risk_band(750), "low")
        self.assertEqual(assign_risk_band(850), "low")
        self.assertEqual(assign_risk_band(650), "medium")
        self.assertEqual(assign_risk_band(749), "medium")
        self.assertEqual(assign_risk_band(649), "high")
        self.assertEqual(assign_risk_band(300), "high")


if __name__ == "__main__":
    unittest.main()
