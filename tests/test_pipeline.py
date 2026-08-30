import unittest

import numpy as np

from reference_calibration.config import SimulationConfig
from reference_calibration.evaluation import calibrate_report
from reference_calibration.experiment import report_specs
from reference_calibration.joint_decoding import (
    JointDecoderConfig,
    calibrate_report_joint,
    calibrate_report_pairwise_maximum_entropy,
)
from reference_calibration.measurement import measure_report
from reference_calibration.models import CalibrationModel, LatentMixtureModel, PairAwareLogModel
from reference_calibration.population import META_CAMPAIGN_SCENARIOS, generate_campaign, make_world
from reference_calibration.measurement import calibration_dataset
from reference_calibration.experiment import calibration_checkpoints


class ConstantCaptureModel(CalibrationModel):
    name = "constant_test_model"

    def predict_capture(self, subset_masks, log_scale):
        del log_scale
        return np.full(len(subset_masks), 0.20)

    def describe(self):
        return {"name": self.name}


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SimulationConfig(n_users=2_000)
        cls.world = make_world(cls.config)
        cls.campaign = generate_campaign(cls.world, "two_small_correlated", 777, "pipeline")

    def test_calibrated_report_has_valid_basic_bounds(self):
        observation = measure_report(
            self.world,
            self.campaign,
            tuple(range(self.config.n_weeks)),
            tuple(range(self.config.n_edps)),
        )
        result = calibrate_report(observation, ConstantCaptureModel())
        marginals = [observation.truth_intersections[1 << i] for i in range(self.config.n_edps)]
        self.assertTrue(np.all(result.exclusive_cells >= 0))
        np.testing.assert_allclose(
            [result.union_values[1 << i] for i in range(self.config.n_edps)],
            marginals,
            rtol=1e-9,
            atol=1e-5,
        )
        self.assertGreaterEqual(result.full_union + 1e-5, max(marginals))
        self.assertLessEqual(result.full_union, self.config.population_size + 1e-5)

    def test_required_report_shapes_are_present(self):
        labels = {label for label, _, _ in report_specs(10, 13)}
        self.assertIn("weeks_1_3__2_edps", labels)
        self.assertIn("weeks_1_12__10_edps", labels)
        self.assertIn("weeks_5_12__2_edps", labels)
        self.assertIn("weeks_7_13__5_edps", labels)
        self.assertIn("all_weeks__10_edps", labels)

    def test_product_facing_scenarios_generate_valid_campaigns(self):
        for index, scenario in enumerate(META_CAMPAIGN_SCENARIOS):
            campaign = generate_campaign(
                self.world,
                scenario,
                10_000 + index,
                f"scenario_{scenario}",
            )
            self.assertEqual(campaign.events.shape, (10, 13, self.config.n_users))
            self.assertTrue(np.all(campaign.final_reach_fraction > 0))
            self.assertTrue(np.all(campaign.final_reach_fraction <= 0.95))

    def test_scenario_sensitivity_overrides_generate_campaigns(self):
        for similarity, matchability in ((0.60, -0.50), (1.40, 0.50)):
            campaign = generate_campaign(
                self.world,
                "website_retargeting",
                20_000 + int(100 * similarity),
                f"sensitivity_{similarity}_{matchability}",
                similarity_multiplier=similarity,
                matchability_shift=matchability,
            )
            self.assertEqual(campaign.events.shape, (10, 13, self.config.n_users))

    def test_pairwise_and_joint_decoders_preserve_marginals(self):
        calibration_campaigns = [
            generate_campaign(self.world, "representative", 30_000 + index, f"fit_{index}")
            for index in range(3)
        ]
        observations = [
            measure_report(
                self.world,
                campaign,
                weeks,
                tuple(range(self.config.n_edps)),
            )
            for campaign in calibration_campaigns
            for weeks in calibration_checkpoints(self.config.n_weeks)
        ]
        data = calibration_dataset(observations, 1.0)
        pair = PairAwareLogModel.fit(data, self.config.n_edps, "shared", 0.35)
        mixture = LatentMixtureModel.fit(data, self.config.n_edps, 40_000)
        observation = measure_report(
            self.world,
            self.campaign,
            tuple(range(self.config.n_weeks)),
            tuple(range(5)),
        )
        results = (
            calibrate_report_pairwise_maximum_entropy(observation, pair),
            calibrate_report_joint(
                observation,
                mixture,
                JointDecoderConfig("joint_test", response_mode="mixture_exact"),
            ),
        )
        expected = np.array(
            [observation.truth_intersections[1 << local] for local in range(5)]
        )
        for result in results:
            actual = np.array([result.union_values[1 << local] for local in range(5)])
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-4)
            self.assertTrue(np.all(result.exclusive_cells >= 0))


if __name__ == "__main__":
    unittest.main()
