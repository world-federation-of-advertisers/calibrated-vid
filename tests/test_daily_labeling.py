from __future__ import annotations

import unittest

import numpy as np

from reference_calibration.config import SimulationConfig
from reference_calibration.daily_labeling import (
    _report_union,
    _truth_union,
    label_campaign_portfolio,
    label_collision_resolved_overlap_pool,
    label_fixed_marginal_overlap_atlas,
    label_hash_pool,
    label_oracle_online_union,
    label_oracle_online_venn,
    label_pair_targeted_fixed_marginal_atlas,
)
from reference_calibration.models import CalibrationModel
from reference_calibration.population import generate_campaign, make_world


class ConstantCaptureModel(CalibrationModel):
    name = "constant_capture"

    def predict_capture(self, subset_masks, log_scale):
        del log_scale
        return np.full(len(subset_masks), 0.45, dtype=float)

    def describe(self):
        return {"name": self.name}


class DailyLabelingTest(unittest.TestCase):
    def setUp(self):
        self.world = make_world(
            SimulationConfig(
                n_users=1_200,
                population_size=1_200_000,
                n_edps=5,
                n_weeks=6,
                seed=7123,
            )
        )
        self.campaign = generate_campaign(
            self.world,
            "website_retargeting",
            seed=9182,
            campaign_id="test_campaign",
        )

    def test_sticky_map_keeps_repeated_identifier_on_one_vid(self):
        result = label_hash_pool(
            self.world,
            self.campaign,
            np.linspace(0.0, 0.9, self.world.config.n_weeks),
            "sticky",
            bridge_pool_fraction=0.05,
            sticky=True,
        )
        for edp in range(self.world.config.n_edps):
            for user in range(self.world.config.n_users):
                values = result.labels[edp, :, user]
                values = values[values >= 0]
                if len(values):
                    self.assertEqual(len(np.unique(values)), 1)

    def test_immutable_labels_make_nested_union_monotone(self):
        result = label_hash_pool(
            self.world,
            self.campaign,
            np.linspace(0.1, 0.8, self.world.config.n_weeks),
            "adaptive",
            bridge_pool_fraction=0.05,
        )
        early = _report_union(result.labels, (0, 1), (0, 1))
        later = _report_union(result.labels, tuple(range(5)), (0, 1, 2, 3))
        self.assertLessEqual(early, later)
        self.assertEqual(early, _report_union(result.labels, (0, 1), (0, 1)))

    def test_oracle_online_venn_matches_cumulative_prefix(self):
        result = label_oracle_online_venn(
            self.world,
            self.campaign,
            edp_count=5,
            prefer_recent_slots=True,
        )
        for day in range(self.world.config.n_weeks):
            weeks = tuple(range(day + 1))
            for edps in ((0, 1), (0, 1, 2), (0, 1, 2, 3, 4)):
                self.assertEqual(
                    _report_union(result.labels, weeks, edps),
                    _truth_union(self.campaign, weeks, edps),
                )

    def test_global_daily_state_is_campaign_order_independent(self):
        second = generate_campaign(
            self.world,
            "crm_customer_list",
            seed=8127,
            campaign_id="second_campaign",
        )
        campaigns = [self.campaign, second]
        dials = {
            self.campaign.campaign_id: np.zeros(self.world.config.n_weeks),
            second.campaign_id: np.ones(self.world.config.n_weeks) * 0.9,
        }
        forward = label_campaign_portfolio(
            self.world,
            campaigns,
            dials,
            "forward",
            bridge_pool_fraction=0.05,
            campaign_order=(0, 1),
            global_day_dial=True,
        )
        reverse = label_campaign_portfolio(
            self.world,
            campaigns,
            dials,
            "reverse",
            bridge_pool_fraction=0.05,
            campaign_order=(1, 0),
            global_day_dial=True,
        )
        for campaign in campaigns:
            np.testing.assert_array_equal(
                forward.labels_by_campaign[campaign.campaign_id],
                reverse.labels_by_campaign[campaign.campaign_id],
            )

    def test_oracle_online_union_matches_full_roster_prefix(self):
        result = label_oracle_online_union(
            self.world,
            self.campaign,
            prefer_recent_slots=True,
        )
        all_edps = tuple(range(self.world.config.n_edps))
        for day in range(self.world.config.n_weeks):
            weeks = tuple(range(day + 1))
            self.assertEqual(
                _report_union(result.labels, weeks, all_edps),
                _truth_union(self.campaign, weeks, all_edps),
            )

    def test_collision_resolved_pool_preserves_each_edp_reach(self):
        result = label_collision_resolved_overlap_pool(
            self.world,
            self.campaign,
            np.linspace(0.0, 0.9, self.world.config.n_weeks),
            bridge_pool_fraction=0.05,
        )
        weeks = tuple(range(self.world.config.n_weeks))
        for edp in range(self.world.config.n_edps):
            self.assertEqual(
                _report_union(result.labels, weeks, (edp,)),
                _truth_union(self.campaign, weeks, (edp,)),
            )

    def test_fixed_marginal_atlas_preserves_each_edp_reach(self):
        result = label_fixed_marginal_overlap_atlas(
            self.world,
            self.campaign,
            np.linspace(0.0, 0.9, self.world.config.n_weeks),
        )
        weeks = tuple(range(self.world.config.n_weeks))
        for edp in range(self.world.config.n_edps):
            self.assertEqual(
                _report_union(result.labels, weeks, (edp,)),
                _truth_union(self.campaign, weeks, (edp,)),
            )

    def test_pair_targeted_atlas_preserves_each_edp_reach(self):
        result = label_pair_targeted_fixed_marginal_atlas(
            self.world,
            self.campaign,
            ConstantCaptureModel(),
            np.linspace(0.0, 0.9, self.world.config.n_weeks),
        )
        weeks = tuple(range(self.world.config.n_weeks))
        for edp in range(self.world.config.n_edps):
            self.assertEqual(
                _report_union(result.labels, weeks, (edp,)),
                _truth_union(self.campaign, weeks, (edp,)),
            )


if __name__ == "__main__":
    unittest.main()
