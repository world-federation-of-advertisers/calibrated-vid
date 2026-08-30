from __future__ import annotations

import unittest

import numpy as np

from reference_calibration.config import SimulationConfig
from reference_calibration.daily_labeling import label_oracle_online_venn
from reference_calibration.population import generate_campaign, make_world
from reference_calibration.venn_information_proof import (
    _label_exact_cells,
    _truth_exact_cells,
    indistinguishable_window_counterexample,
    label_daily_full_venn,
    time_atom_audit,
)


class FullVennInformationTest(unittest.TestCase):
    def setUp(self):
        self.world = make_world(
            SimulationConfig(
                n_users=800,
                population_size=800_000,
                n_edps=10,
                n_weeks=5,
                seed=8801,
            )
        )
        self.campaign = generate_campaign(
            self.world,
            "website_retargeting",
            seed=8802,
            campaign_id="full_venn_test",
        )

    def test_daily_and_cumulative_counts_do_not_identify_middle_window(self):
        example = indistinguishable_window_counterexample()
        self.assertTrue(example["daily_equal"])
        self.assertTrue(example["cumulative_equal"])
        self.assertNotEqual(
            example["world_1_weeks_2_3_union"],
            example["world_2_weeks_2_3_union"],
        )

    def test_daily_full_venn_is_exact_for_each_day(self):
        result = label_daily_full_venn(self.campaign)
        edps = tuple(range(self.world.config.n_edps))
        for day in range(self.world.config.n_weeks):
            np.testing.assert_array_equal(
                _label_exact_cells(result.labels, (day,), edps)[1:],
                _truth_exact_cells(self.campaign, (day,), edps)[1:],
            )

    def test_ten_edp_cumulative_full_venn_is_exact_for_every_prefix_cell(self):
        result = label_oracle_online_venn(
            self.world,
            self.campaign,
            edp_count=10,
            prefer_recent_slots=True,
        )
        edps = tuple(range(self.world.config.n_edps))
        for day in range(self.world.config.n_weeks):
            weeks = tuple(range(day + 1))
            np.testing.assert_array_equal(
                _label_exact_cells(result.labels, weeks, edps)[1:],
                _truth_exact_cells(self.campaign, weeks, edps)[1:],
            )

    def test_time_activity_atoms_answer_every_supported_query(self):
        audit = time_atom_audit(self.campaign, n_edps=3, n_weeks=4)
        self.assertEqual(audit["report_queries_checked"], 105)
        self.assertEqual(audit["max_report_difference"], 0)


if __name__ == "__main__":
    unittest.main()
