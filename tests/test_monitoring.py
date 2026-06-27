import unittest

import numpy as np

from src.monitoring import population_stability_index


class MonitoringTest(unittest.TestCase):
    def test_psi_near_zero_for_identical_distributions(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(size=10000)
        self.assertLess(population_stability_index(sample, sample), 0.01)

    def test_psi_large_for_shifted_distribution(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, 10000)
        current = rng.normal(2, 1, 10000)  # forte deslocamento
        self.assertGreater(population_stability_index(reference, current), 0.25)

    def test_psi_increases_with_drift(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, 10000)
        small = population_stability_index(reference, rng.normal(0.2, 1, 10000))
        big = population_stability_index(reference, rng.normal(1.0, 1, 10000))
        self.assertGreater(big, small)


if __name__ == "__main__":
    unittest.main()
