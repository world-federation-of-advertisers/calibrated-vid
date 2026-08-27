import unittest

import numpy as np

from reference_calibration.config import SimulationConfig
from reference_calibration.measurement import measure_report
from reference_calibration.population import DEMOGRAPHIC_LABELS, generate_campaign, make_world
from reference_calibration.provider_model import (
    ContextualDemographicAllocator,
    FixedDemographicAllocator,
    PanelTotalReachModel,
    ProportionalDemographicAllocator,
)


class ProviderModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SimulationConfig(n_users=2_000)
        cls.world = make_world(cls.config)
        scenarios = (
            "broad_awareness_control",
            "website_retargeting",
            "crm_customer_list",
            "app_activity_retargeting",
        )
        cls.observations = []
        for index, scenario in enumerate(scenarios):
            campaign = generate_campaign(
                cls.world,
                scenario,
                cls.config.seed + 80_000 + index,
                f"provider_fit_{index}",
            )
            for edps in ((0, 1), tuple(range(5)), tuple(range(10))):
                cls.observations.append(
                    measure_report(
                        cls.world,
                        campaign,
                        tuple(range(cls.config.n_weeks)),
                        edps,
                    )
                )

    def test_world_and_observation_include_demographics(self):
        self.assertEqual(len(DEMOGRAPHIC_LABELS), 18)
        self.assertAlmostEqual(
            float(self.world.true_demographic_population.sum()),
            self.config.population_size,
        )
        observation = self.observations[0]
        self.assertEqual(len(observation.objectives), len(observation.edps))
        self.assertEqual(len(observation.audience_strategies), len(observation.edps))
        self.assertAlmostEqual(
            float(observation.truth_demographic_union.sum()),
            float(observation.truth_unions[-1]),
            places=5,
        )
        self.assertAlmostEqual(
            float(observation.baseline_demographic_union.sum()),
            float(observation.baseline_unions[-1]),
            places=5,
        )

    def test_panel_total_model_is_bounded(self):
        for include_context in (False, True):
            model = PanelTotalReachModel.fit(
                self.observations,
                self.config.n_edps,
                include_context,
            )
            for observation in self.observations:
                estimate = model.predict(observation)
                marginals = np.asarray(
                    [
                        observation.truth_intersections[1 << local]
                        for local in range(len(observation.edps))
                    ]
                )
                self.assertGreaterEqual(estimate, float(np.max(marginals)) - 1e-5)
                self.assertLessEqual(
                    estimate,
                    min(float(np.sum(marginals)), self.config.population_size) + 1e-5,
                )

    def test_demographic_allocators_match_total_and_population_bounds(self):
        allocators = (
            ProportionalDemographicAllocator(),
            FixedDemographicAllocator.fit(self.observations),
            ContextualDemographicAllocator.fit(
                self.observations,
                self.config.n_edps,
            ),
        )
        for observation in self.observations:
            total = float(observation.truth_unions[-1])
            for allocator in allocators:
                result = allocator.allocate(total, observation)
                self.assertAlmostEqual(float(result.sum()), total, places=4)
                self.assertTrue(np.all(result >= -1e-6))
                self.assertTrue(
                    np.all(result <= observation.demographic_population + 1e-5)
                )


if __name__ == "__main__":
    unittest.main()
