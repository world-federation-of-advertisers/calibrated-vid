from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from calibrated_vid.behavioral_market import (
    behavioral_diagnostics,
    build_audience,
    fit_overlap_model,
    make_campaign_plans,
    predict_campaign,
    run_behavioral_experiment,
    simulate_campaign,
    simulate_market,
    summarize_scenarios,
)


class BehavioralMarketTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.campaigns = simulate_market()
        cls.model = fit_overlap_model(cls.campaigns)
        cls.predictions = [
            predict_campaign(campaign, cls.model) for campaign in cls.campaigns
        ]

    def test_balanced_scenarios_are_generated(self) -> None:
        self.assertEqual(len(self.campaigns), 328)
        evaluation_counts: dict[str, int] = {}
        for campaign in self.campaigns:
            if campaign.split == "evaluation":
                evaluation_counts[campaign.scenario] = (
                    evaluation_counts.get(campaign.scenario, 0) + 1
                )
        self.assertEqual(
            evaluation_counts,
            {
                "conversion_large_large": 20,
                "reach_large_large": 48,
                "reach_large_medium": 32,
                "reach_medium_large": 32,
                "traffic_large_large": 32,
            },
        )

    def test_traffic_delivery_has_lower_overlap_than_matched_reach_delivery(self) -> None:
        diagnostic = behavioral_diagnostics(self.campaigns)["paired_objective_test"]
        self.assertGreaterEqual(diagnostic["traffic_lower_share"], 0.80)
        self.assertLess(diagnostic["traffic_minus_reach_median_points"], -10.0)

    def test_objective_pairs_hold_campaign_conditions_constant(self) -> None:
        pairs: dict[str, dict[str, object]] = {}
        for campaign in self.campaigns:
            if campaign.scenario in {"reach_large_large", "traffic_large_large"}:
                pairs.setdefault(campaign.comparison_id, {})[campaign.objective] = campaign
        for rows in pairs.values():
            if set(rows) != {"reach", "traffic"}:
                continue
            reach = rows["reach"]
            traffic = rows["traffic"]
            self.assertEqual(reach.split, traffic.split)
            self.assertEqual(reach.reach_a, traffic.reach_a)
            self.assertEqual(reach.reach_b, traffic.reach_b)
            self.assertEqual(reach.active_days, traffic.active_days)
            self.assertEqual(reach.short_video_share_a, traffic.short_video_share_a)
            self.assertEqual(reach.short_video_share_b, traffic.short_video_share_b)
            self.assertEqual(reach.auction_state, traffic.auction_state)
            self.assertEqual(
                reach.opportunity_asymmetry,
                traffic.opportunity_asymmetry,
            )

    def test_reversing_publisher_size_changes_overlap(self) -> None:
        diagnostic = behavioral_diagnostics(self.campaigns)["paired_direction_test"]
        self.assertGreaterEqual(diagnostic["medium_large_higher_share"], 0.90)
        self.assertGreater(
            diagnostic["medium_large_minus_large_medium_median_points"], 12.0
        )

    def test_direction_pairs_only_swap_publisher_sizes(self) -> None:
        pairs: dict[str, dict[str, object]] = {}
        for campaign in self.campaigns:
            if campaign.scenario in {"reach_medium_large", "reach_large_medium"}:
                pairs.setdefault(campaign.comparison_id, {})[campaign.direction] = campaign
        for rows in pairs.values():
            medium_large = rows["medium_large"]
            large_medium = rows["large_medium"]
            self.assertEqual(medium_large.split, large_medium.split)
            self.assertEqual(medium_large.reach_a, large_medium.reach_b)
            self.assertEqual(medium_large.reach_b, large_medium.reach_a)
            self.assertEqual(medium_large.active_days, large_medium.active_days)
            self.assertEqual(
                medium_large.short_video_share_a,
                large_medium.short_video_share_a,
            )
            self.assertEqual(
                medium_large.short_video_share_b,
                large_medium.short_video_share_b,
            )
            self.assertEqual(medium_large.auction_state, large_medium.auction_state)
            self.assertEqual(
                medium_large.opportunity_asymmetry,
                large_medium.opportunity_asymmetry,
            )

    def test_large_reach_campaigns_retain_material_dispersion(self) -> None:
        diagnostic = behavioral_diagnostics(self.campaigns)["within_large_reach"]
        self.assertGreater(
            diagnostic["overlap_p10_p90_width_points"],
            20.0,
        )

    def test_campaign_id_does_not_change_the_sampling_draw(self) -> None:
        audience = build_audience()
        plan = next(
            plan
            for plan in make_campaign_plans()
            if plan.scenario == "reach_large_large"
        )
        renamed = replace(plan, campaign_id="renamed-campaign")
        self.assertEqual(
            simulate_campaign(audience, plan),
            replace(
                simulate_campaign(audience, renamed),
                campaign_id=plan.campaign_id,
            ),
        )

    def test_reference_signal_improves_every_scenario(self) -> None:
        summary = summarize_scenarios(self.campaigns, self.predictions)
        for row in summary:
            with self.subTest(scenario=row["scenario"]):
                self.assertLess(
                    row["calibrated_overlap_mae_points"],
                    row["fixed_overlap_mae_points"],
                )
                self.assertLess(
                    row["calibrated_union_mape_percent"],
                    row["fixed_union_mape_percent"],
                )
                self.assertLess(
                    row["calibrated_incremental_mape_percent"],
                    row["fixed_incremental_mape_percent"],
                )

    def test_predictions_preserve_publisher_reach(self) -> None:
        by_id = {campaign.campaign_id: campaign for campaign in self.campaigns}
        for prediction in self.predictions:
            campaign = by_id[prediction.campaign_id]
            fixed_overlap = (
                campaign.reach_a + campaign.reach_b - prediction.fixed_union
            )
            calibrated_overlap = (
                campaign.reach_a + campaign.reach_b - prediction.calibrated_union
            )
            self.assertGreaterEqual(fixed_overlap, 0.0)
            self.assertGreaterEqual(calibrated_overlap, 0.0)
            self.assertLessEqual(fixed_overlap, min(campaign.reach_a, campaign.reach_b))
            self.assertLessEqual(
                calibrated_overlap, min(campaign.reach_a, campaign.reach_b)
            )
            smaller_reach = min(campaign.reach_a, campaign.reach_b)
            self.assertAlmostEqual(
                prediction.fixed_incremental_unique_reach,
                smaller_reach - fixed_overlap,
            )
            self.assertAlmostEqual(
                prediction.calibrated_incremental_unique_reach,
                smaller_reach - calibrated_overlap,
            )

    def test_fit_ignores_evaluation_labels(self) -> None:
        changed = [
            replace(
                campaign,
                overlap_rate=0.99,
                fingerprint_overlap_estimate=0.01,
            )
            if campaign.split == "evaluation"
            else campaign
            for campaign in self.campaigns
        ]
        self.assertEqual(fit_overlap_model(changed), self.model)

    def test_outputs_are_reproducible(self) -> None:
        filenames = ["segments.csv", "campaigns.csv", "predictions.csv", "summary.json"]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            run_behavioral_experiment(first)
            run_behavioral_experiment(second)
            for filename in filenames:
                self.assertEqual(
                    (Path(first) / filename).read_bytes(),
                    (Path(second) / filename).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
