from __future__ import annotations

import unittest

import numpy as np

from reference_calibration.calibrated_venn_labeling import (
    label_from_cumulative_targets,
    reconcile_reachable_cells_greedy,
)
from reference_calibration.config import SimulationConfig
from reference_calibration.daily_labeling import _report_union, _transport_cells, _truth_union
from reference_calibration.population import generate_campaign, make_world
from reference_calibration.venn_information_proof import (
    _label_exact_cells,
    _truth_exact_cells,
)


class CalibratedVennLabelingTest(unittest.TestCase):
    def setUp(self):
        self.world = make_world(
            SimulationConfig(
                n_users=600,
                population_size=600_000,
                n_edps=5,
                n_weeks=5,
                seed=9911,
            )
        )
        self.campaign = generate_campaign(
            self.world,
            "website_retargeting",
            seed=9912,
            campaign_id="calibrated_venn_test",
        )

    def test_greedy_temporal_projection_preserves_marginals_and_reachability(self):
        current = np.zeros(1 << self.world.config.n_edps, dtype=int)
        current[0] = self.world.config.n_users
        for day in range(self.world.config.n_weeks):
            truth = _truth_exact_cells(
                self.campaign,
                tuple(range(day + 1)),
                tuple(range(self.world.config.n_edps)),
            ).astype(float)
            raw = truth.copy()
            raw[0] += 0.35
            raw[-1] = max(raw[-1] - 0.35, 0.0)
            marginals = np.asarray(
                [
                    sum(truth[mask] for mask in range(len(truth)) if mask & (1 << edp))
                    for edp in range(self.world.config.n_edps)
                ],
                dtype=int,
            )
            reconciled = reconcile_reachable_cells_greedy(current, raw, marginals)
            _transport_cells(current, reconciled.target_cells)
            for edp, marginal in enumerate(marginals):
                estimate = sum(
                    reconciled.target_cells[mask]
                    for mask in range(len(truth))
                    if mask & (1 << edp)
                )
                self.assertEqual(estimate, marginal)
            current = reconciled.target_cells

    def test_reachable_true_targets_produce_exact_prefix_cells(self):
        edps = tuple(range(self.world.config.n_edps))
        targets = [
            _truth_exact_cells(self.campaign, tuple(range(day + 1)), edps)
            for day in range(self.world.config.n_weeks)
        ]
        result = label_from_cumulative_targets(
            self.campaign,
            targets,
            "truth_targets",
            timing_policy="active_today",
        )
        for day, target in enumerate(targets):
            np.testing.assert_array_equal(
                _label_exact_cells(result.labels, tuple(range(day + 1)), edps)[1:],
                target[1:],
            )
        for edp in edps:
            for week_mask in range(1, 1 << self.world.config.n_weeks):
                weeks = tuple(
                    week
                    for week in range(self.world.config.n_weeks)
                    if week_mask & (1 << week)
                )
                self.assertEqual(
                    _report_union(result.labels, weeks, (edp,)),
                    _truth_union(self.campaign, weeks, (edp,)),
                )

if __name__ == "__main__":
    unittest.main()
