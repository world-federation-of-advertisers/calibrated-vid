from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from calibrated_vid.intuitive_market import (
    PROFILES,
    IntuitiveConfig,
    _blend_positive_pair,
    _effective_active_days,
    build_population,
    diagnostics,
    run_experiment,
    simulate_market,
)


class IntuitiveMarketTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = IntuitiveConfig(
            population_size=60_000,
            medium_reach=3_600,
            large_reach=12_000,
            training_replicates=6,
            objective_replicates_per_profile=3,
            direction_replicates=8,
            large_profile_replicates=4,
        )
        cls.report = diagnostics(simulate_market(cls.config))

    def test_intuitive_mechanisms_reproduce_three_failure_shapes(self) -> None:
        self.assertEqual(self.report["objective"]["traffic_lower_share"], 1.0)
        self.assertLess(
            self.report["objective"]["traffic_minus_reach_median_points"],
            -8.0,
        )
        self.assertEqual(self.report["direction"]["medium_large_higher_share"], 1.0)
        self.assertGreater(
            self.report["direction"][
                "medium_large_minus_large_medium_median_points"
            ],
            10.0,
        )
        self.assertGreater(
            self.report["large_profiles"]["p10_p90_width_points"],
            15.0,
        )

    def test_direction_disappears_without_publisher_activity_contrast(self) -> None:
        report = diagnostics(
            simulate_market(replace(self.config, activity_contrast=0.0))
        )
        self.assertLess(
            abs(
                report["direction"][
                    "medium_large_minus_large_medium_median_points"
                ]
            ),
            2.0,
        )

    def test_objective_order_reverses_without_publisher_response_contrast(self) -> None:
        report = diagnostics(
            simulate_market(replace(self.config, response_contrast=0.0))
        )
        self.assertGreater(
            report["objective"]["traffic_minus_reach_median_points"],
            0.0,
        )

    def test_large_reach_dispersion_collapses_when_profiles_are_identical(self) -> None:
        report = diagnostics(
            simulate_market(replace(self.config, profile_strength=0.0))
        )
        self.assertLess(
            report["large_profiles"]["p10_p90_width_points"],
            2.0,
        )

    def test_positive_cost_blending_preserves_unbounded_inputs(self) -> None:
        publisher_a = np.asarray([1.40, 0.80])
        publisher_b = np.asarray([1.10, 2.20])

        unchanged_a, unchanged_b = _blend_positive_pair(
            publisher_a,
            publisher_b,
            1.0,
        )
        np.testing.assert_allclose(unchanged_a, publisher_a)
        np.testing.assert_allclose(unchanged_b, publisher_b)

        midpoint_a, midpoint_b = _blend_positive_pair(
            publisher_a,
            publisher_b,
            0.0,
        )
        expected_midpoint = 0.5 * (publisher_a + publisher_b)
        np.testing.assert_allclose(midpoint_a, expected_midpoint)
        np.testing.assert_allclose(midpoint_b, expected_midpoint)
        self.assertTrue(np.all(midpoint_a > 0.0))
        self.assertTrue(np.all(midpoint_b > 0.0))

        extrapolated_a, extrapolated_b = _blend_positive_pair(
            np.asarray([0.10]),
            np.asarray([10.0]),
            1.5,
        )
        self.assertTrue(np.all(extrapolated_a > 0.0))
        self.assertTrue(np.all(extrapolated_b > 0.0))

    def test_profile_strength_boundary_preserves_positive_durations(self) -> None:
        config = replace(self.config, profile_strength=1.25)
        build_population(config)
        self.assertTrue(
            all(
                _effective_active_days(profile, config.profile_strength) > 0.0
                for profile in PROFILES
            )
        )

    def test_profile_strength_above_validated_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "profile_strength must be between 0 and 1.25",
        ):
            build_population(replace(self.config, profile_strength=1.5))

    def test_outputs_are_reproducible(self) -> None:
        filenames = ["segments.csv", "profiles.csv", "campaigns.csv", "summary.json"]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            run_experiment(first, self.config)
            run_experiment(second, self.config)
            for filename in filenames:
                self.assertEqual(
                    (Path(first) / filename).read_bytes(),
                    (Path(second) / filename).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
