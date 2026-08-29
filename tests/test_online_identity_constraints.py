from __future__ import annotations

import unittest

from reference_calibration.online_identity_constraints import two_day_anchor_counterexample


class OnlineIdentityConstraintsTest(unittest.TestCase):
    def test_two_day_example_exposes_the_online_conflict(self):
        result = two_day_anchor_counterexample()
        self.assertEqual(result["day_1"]["required_union"], 1)
        self.assertEqual(result["day_2"]["required_edp_a_reach"], 2)
        self.assertIn("breaks the shared-email anchor", result["conditional_conflict"])


if __name__ == "__main__":
    unittest.main()
