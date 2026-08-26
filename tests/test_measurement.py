import unittest

import numpy as np

from reference_calibration.config import SimulationConfig
from reference_calibration.measurement import measure_report
from reference_calibration.population import generate_campaign, make_world


class MeasurementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SimulationConfig(n_users=2_000, calibration_train_campaigns=1, calibration_holdout_campaigns=1)
        cls.world = make_world(cls.config)
        cls.campaign = generate_campaign(cls.world, "representative", 42, "test")

    def test_full_venn_export(self):
        observation = measure_report(
            self.world,
            self.campaign,
            tuple(range(self.config.n_weeks)),
            tuple(range(self.config.n_edps)),
        )
        self.assertEqual(len(observation.global_masks) - 1 - self.config.n_edps, 1_013)
        self.assertEqual(len(observation.truth_intersections), 1_024)
        self.assertTrue(np.all(observation.collision_floor >= 0))

    def test_same_subset_has_same_measurement_inside_larger_report(self):
        weeks = tuple(range(self.config.n_weeks))
        pair = measure_report(self.world, self.campaign, weeks, (0, 1))
        full = measure_report(
            self.world,
            self.campaign,
            weeks,
            tuple(range(self.config.n_edps)),
        )
        pair_mask_in_full = (1 << 0) | (1 << 1)
        self.assertEqual(pair.truth_intersections[-1], full.truth_intersections[pair_mask_in_full])
        self.assertEqual(pair.baseline_intersections[-1], full.baseline_intersections[pair_mask_in_full])
        self.assertEqual(pair.reference_intersections[-1], full.reference_intersections[pair_mask_in_full])
        self.assertEqual(pair.reference_signal[-1], full.reference_signal[pair_mask_in_full])


if __name__ == "__main__":
    unittest.main()
