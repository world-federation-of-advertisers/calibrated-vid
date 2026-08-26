import unittest

import numpy as np

from reference_calibration.sets import (
    decode_nonnegative_cells,
    enforce_marginals,
    inclusive_intersections,
    union_values,
)


class SetMathTest(unittest.TestCase):
    def test_nonnegative_decode_recovers_valid_cells(self):
        exact = np.array([0.0, 10.0, 20.0, 5.0, 30.0, 4.0, 3.0, 2.0])
        intersections = inclusive_intersections(exact)
        decoded, residual = decode_nonnegative_cells(intersections)
        np.testing.assert_allclose(decoded[1:], exact[1:], atol=1e-5)
        self.assertLess(residual, 1e-8)
        unions = union_values(decoded)
        self.assertGreaterEqual(unions[0b111], unions[0b011])
        self.assertGreaterEqual(unions[0b011], unions[0b001])

    def test_marginal_projection_holds_each_edp_reach_fixed(self):
        cells = np.array([0.0, 30.0, 20.0, 10.0, 15.0, 8.0, 6.0, 4.0])
        marginals = np.array([42.0, 38.0, 33.0])
        projected = enforce_marginals(cells, marginals, population=100.0)
        intersections = inclusive_intersections(projected)
        np.testing.assert_allclose(
            [intersections[1], intersections[2], intersections[4]],
            marginals,
            rtol=1e-9,
            atol=1e-7,
        )
        self.assertAlmostEqual(projected.sum(), 100.0, places=6)

    def test_marginal_projection_fallback_handles_ten_edps(self):
        population = 1_000_000.0
        cells = np.linspace(1.0, 2.0, 1 << 10)
        marginals = population * np.array(
            [0.02, 0.08, 0.15, 0.28, 0.42, 0.57, 0.71, 0.84, 0.92, 0.97]
        )
        projected = enforce_marginals(
            cells,
            marginals,
            population=population,
            max_iterations=0,
            tolerance=1e-8,
        )
        intersections = inclusive_intersections(projected)
        np.testing.assert_allclose(
            [intersections[1 << edp] for edp in range(10)],
            marginals,
            rtol=1e-8,
            atol=1e-3,
        )
        self.assertAlmostEqual(projected.sum(), population, places=4)


if __name__ == "__main__":
    unittest.main()
