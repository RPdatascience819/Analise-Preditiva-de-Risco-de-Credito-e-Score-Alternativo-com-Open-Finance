import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.train import save_model, split_features_target, train_baseline_classifier


class TrainTest(unittest.TestCase):
    def test_split_features_target(self):
        df = pd.DataFrame(
            {
                "feature_a": [1, 2, 3],
                "feature_b": [10, 20, 30],
                "TARGET": [0, 1, 0],
            }
        )

        X, y = split_features_target(df)

        self.assertNotIn("TARGET", X.columns)
        self.assertEqual(list(X.columns), ["feature_a", "feature_b"])
        self.assertEqual(y.tolist(), [0, 1, 0])

    def test_train_baseline_classifier_returns_fitted_model_and_split(self):
        df = pd.DataFrame(
            {
                "feature_a": list(range(20)),
                "feature_b": [value * 2 for value in range(20)],
                "TARGET": [0, 1] * 10,
            }
        )

        model, X_train, X_test, y_train, y_test = train_baseline_classifier(df, test_size=0.25)

        self.assertIsInstance(model, RandomForestClassifier)
        self.assertTrue(hasattr(model, "estimators_"))
        self.assertEqual(len(X_train), 15)
        self.assertEqual(len(X_test), 5)
        self.assertEqual(len(y_train), 15)
        self.assertEqual(len(y_test), 5)
        self.assertNotIn("TARGET", X_train.columns)
        self.assertNotIn("TARGET", X_test.columns)

    def test_save_model_creates_parent_directory_and_file(self):
        df = pd.DataFrame(
            {
                "feature_a": list(range(20)),
                "feature_b": [value * 2 for value in range(20)],
                "TARGET": [0, 1] * 10,
            }
        )
        model, *_ = train_baseline_classifier(df, test_size=0.25)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "nested" / "model.joblib"

            save_model(model, model_path)

            self.assertTrue(model_path.is_file())


if __name__ == "__main__":
    unittest.main()
