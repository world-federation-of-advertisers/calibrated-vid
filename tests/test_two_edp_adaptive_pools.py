from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reference_calibration.two_edp_adaptive_pools import (
    EDPS,
    TwoEdpAdaptiveAllocator,
    coverage_feasibility,
    example_days,
    run_example,
)


class TwoEdpAdaptivePoolTest(unittest.TestCase):
    def setUp(self):
        self.events, self.targets = example_days()
        self.allocator = TwoEdpAdaptiveAllocator()
        for day in sorted(self.targets):
            self.allocator.assign_day(
                day,
                [event for event in self.events if event.day == day],
                self.targets[day],
            )

    def test_same_reference_id_gets_same_vid(self):
        alice_a = self.allocator.account_rank[("EDP_A", "a-alice")]
        alice_b = self.allocator.account_rank[("EDP_B", "b-alice")]
        bob_a = self.allocator.account_rank[("EDP_A", "a-bob")]
        bob_b = self.allocator.account_rank[("EDP_B", "b-bob")]
        self.assertEqual(alice_a, alice_b)
        self.assertEqual(bob_a, bob_b)

    def test_proprietary_fallback_can_supply_calibrated_overlap(self):
        self.assertEqual(
            self.allocator.account_rank[("EDP_A", "a-carol")],
            self.allocator.account_rank[("EDP_B", "b-carol")],
        )

    def test_single_publisher_ranks_remain_unique(self):
        for edp in EDPS:
            ranks = [
                rank
                for (account_edp, _), rank in self.allocator.account_rank.items()
                if account_edp == edp
            ]
            self.assertEqual(len(ranks), len(set(ranks)))

    def test_lower_target_does_not_remap_prior_assignments(self):
        day_2 = self.allocator.day_summaries[1]
        day_3 = self.allocator.day_summaries[2]
        self.assertEqual(day_2.achieved_overlap, 4)
        self.assertEqual(day_3.requested_overlap, 3)
        self.assertEqual(day_3.achieved_overlap, 4)
        self.assertEqual(day_3.target_status, "PROJECTED_UP_TO_IMMUTABLE_LOWER_BOUND")

    def test_late_reference_conflict_preserves_existing_vid(self):
        heidi_a = self.allocator.account_rank[("EDP_A", "a-heidi")]
        heidi_b = self.allocator.account_rank[("EDP_B", "b-heidi")]
        self.assertNotEqual(heidi_a, heidi_b)
        self.assertEqual(self.allocator.reference_rank[404], heidi_b)
        self.assertEqual(self.allocator.day_summaries[-1].late_anchor_conflicts, 1)

    def test_unequal_email_coverage_can_be_feasible(self):
        result = coverage_feasibility(0.10, 0.90, 0.60)
        self.assertTrue(result.feasible)
        self.assertEqual(result.shortfall, 0)

    def test_high_coverage_with_low_agreement_can_exhaust_flexible_supply(self):
        result = coverage_feasibility(0.90, 0.90, 0.60)
        self.assertFalse(result.feasible)
        self.assertGreater(result.shortfall, 0)

    def test_example_writes_model_manifests_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_example(output)
            self.assertTrue((output / "two_edp_ranked_model.textproto").exists())
            self.assertTrue((output / "adaptive_allocation.proto").exists())
            self.assertTrue((output / "day_01_allocation_manifest.textproto").exists())
            self.assertTrue((output / "labeler_inputs" / "a-imp-001.textproto").exists())
            self.assertTrue((output / "email_coverage_feasibility.csv").exists())
            self.assertTrue((output / "WALKTHROUGH.md").exists())


if __name__ == "__main__":
    unittest.main()
