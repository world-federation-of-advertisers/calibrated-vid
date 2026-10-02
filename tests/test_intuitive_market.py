from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from calibrated_vid.intuitive_market import (
    IntuitiveConfig,
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
