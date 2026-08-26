import unittest

from reference_calibration.reconciliation import (
    FinalizedReport,
    ReportCandidate,
    ReportKey,
    ResultRegistry,
)


class ReconciliationTest(unittest.TestCase):
    def test_exact_repeat_is_identical(self):
        registry = ResultRegistry(13)
        candidate = ReportCandidate(
            ReportKey("campaign", "model", (0, 1, 2), (0, 1)),
            raw_union=70.0,
            marginal_reaches=(50.0, 45.0),
            uncertainty=2.0,
            population_size=100.0,
        )
        first = registry.finalize(candidate)
        second = registry.finalize(candidate)
        self.assertEqual(first, second)

    def test_infeasible_report_is_produced_and_flagged(self):
        registry = ResultRegistry(13)
        small_key = ReportKey("campaign", "model", (0,), (0,))
        large_key = ReportKey("campaign", "model", (0, 1), (0, 1))
        registry.inject_finalized(
            FinalizedReport(small_key, 80.0, 80.0, "OK", 0.0, 100.0, 0.0, 0.0, 0)
        )
        registry.inject_finalized(
            FinalizedReport(large_key, 50.0, 50.0, "OK", 0.0, 100.0, 0.0, 0.0, 0)
        )
        candidate = ReportCandidate(
            ReportKey("campaign", "model", (0,), (0, 1)),
            raw_union=60.0,
            marginal_reaches=(40.0, 35.0),
            uncertainty=5.0,
            population_size=100.0,
        )
        result = registry.finalize(candidate)
        self.assertEqual(result.status, "REVIEW_REQUIRED")
        self.assertGreater(result.slack, 0.0)
        self.assertGreaterEqual(result.finalized_union, 40.0)
        self.assertLessEqual(result.finalized_union, 75.0)
        self.assertEqual(registry.records[0].finalized_union, 80.0)
        self.assertEqual(registry.records[1].finalized_union, 50.0)

    def test_feasible_report_is_moved_to_prior_bound(self):
        registry = ResultRegistry(13)
        prior = ReportCandidate(
            ReportKey("campaign", "model", (0,), (0,)),
            raw_union=60.0,
            marginal_reaches=(60.0,),
            uncertainty=2.0,
            population_size=100.0,
        )
        registry.finalize(prior)
        candidate = ReportCandidate(
            ReportKey("campaign", "model", (0,), (0, 1)),
            raw_union=58.0,
            marginal_reaches=(40.0, 30.0),
            uncertainty=2.0,
            population_size=100.0,
        )
        result = registry.finalize(candidate)
        self.assertEqual(result.status, "RECONCILED")
        self.assertEqual(result.finalized_union, 60.0)
        self.assertEqual(result.slack, 0.0)


if __name__ == "__main__":
    unittest.main()
