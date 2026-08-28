import unittest

import numpy as np

from reference_calibration.config import SimulationConfig
from reference_calibration.panel_validation import (
    PANEL_DESIGNS,
    PanelPairResponseModel,
    draw_panel,
    measure_panel_report,
)
from reference_calibration.population import generate_campaign, make_world


class PanelValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SimulationConfig(n_users=6_000, panel_size=5_000)
        cls.world = make_world(cls.config)
        cls.campaigns = [
            generate_campaign(
                cls.world,
                scenario,
                cls.config.seed + 900_000 + index,
                f"panel_test_{scenario}",
            )
            for index, scenario in enumerate(
                (
                    "broad_awareness_control",
                    "website_retargeting",
                    "crm_customer_list",
                    "app_activity_retargeting",
                )
            )
        ]

    def test_panel_designs_have_requested_size_and_finite_weights(self):
        for index, design in enumerate(PANEL_DESIGNS):
            panel = draw_panel(self.world, design, self.config.seed + index)
            self.assertEqual(panel.raw_size, 5_000)
            self.assertTrue(np.all(np.isfinite(panel.weights)))
            self.assertAlmostEqual(
                float(panel.weights.sum()),
                float(self.config.population_size),
                places=4,
            )
            self.assertGreater(panel.effective_size, 0)
            self.assertLessEqual(panel.effective_size, panel.raw_size + 1e-6)

    def test_panel_pair_model_produces_bounded_report(self):
        panel = draw_panel(self.world, "representative", self.config.seed + 10)
        observations = [
            measure_panel_report(
                self.world,
                campaign,
                panel,
                tuple(range(self.config.n_weeks)),
                tuple(range(5)),
            )
            for campaign in self.campaigns
        ]
        model = PanelPairResponseModel.fit(observations, self.config.n_edps)
        full = measure_panel_report(
            self.world,
            self.campaigns[0],
            panel,
            tuple(range(self.config.n_weeks)),
            tuple(range(5)),
        )
        result = model.predict_report(full)
        marginals = np.asarray(
            [full.truth_intersections[1 << index] for index in range(5)],
            dtype=float,
        )
        self.assertGreaterEqual(result.full_union, float(np.max(marginals)) - 1e-5)
        self.assertLessEqual(
            result.full_union,
            min(float(np.sum(marginals)), self.config.population_size) + 1e-5,
        )


if __name__ == "__main__":
    unittest.main()
