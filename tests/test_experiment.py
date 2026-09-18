from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from calibrated_vid.experiment import (
    _simulate_campaign,
    fit_models,
    load_experiment,
    predict,
    run_experiment,
    simulate_campaigns,
    summarize,
    validate_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs" / "two_population_experiment.textproto"


class FixedGenerator:
    def __init__(
        self,
        multinomial_results: list[list[int]],
        fingerprint_counts: list[int],
        normal_value: float | None = None,
    ) -> None:
        self.multinomial_results = iter(multinomial_results)
        self.fingerprint_counts = iter(fingerprint_counts)
        self.normal_value = normal_value
        self.observed_probabilities = []

    def normal(self, mean: float, standard_deviation: float) -> float:
        return mean if self.normal_value is None else self.normal_value

    def multinomial(self, population_size: int, probabilities) -> list[int]:
        self.observed_probabilities.append(probabilities)
        return next(self.multinomial_results)

    def binomial(self, population_size: int, probability: float) -> int:
        return next(self.fingerprint_counts)


class ExperimentTest(unittest.TestCase):
    def test_spec_generates_expected_campaign_splits(self) -> None:
        campaigns = simulate_campaigns(load_experiment(SPEC))
        self.assertEqual(len(campaigns), 30)
        self.assertEqual(sum(row.split == "train" for row in campaigns), 10)
        self.assertEqual(
            sum(row.split == "evaluation" and row.objective == "reach" for row in campaigns),
            10,
        )
        self.assertEqual(
            sum(row.split == "evaluation" and row.objective == "traffic" for row in campaigns),
            10,
        )

    def test_fixed_reach_model_misses_traffic_and_reference_signal_corrects_it(self) -> None:
        campaigns = simulate_campaigns(load_experiment(SPEC))
        model = fit_models(campaigns)
        rows = summarize(predict(campaign, model) for campaign in campaigns)
        by_objective = {row["objective"]: row for row in rows}
        self.assertGreater(by_objective["traffic"]["fixed_overlap_mae_points"], 20.0)
        self.assertLess(by_objective["traffic"]["calibrated_overlap_mae_points"], 2.0)
        self.assertLess(
            by_objective["traffic"]["calibrated_union_mape_percent"],
            by_objective["traffic"]["fixed_union_mape_percent"] / 5.0,
        )

    def test_fit_models_ignores_evaluation_campaign_labels(self) -> None:
        campaigns = simulate_campaigns(load_experiment(SPEC))
        expected = fit_models(campaigns)
        modified = [
            replace(
                campaign,
                true_overlap_rate=0.99 if index % 2 == 0 else 0.01,
                fingerprint_inferred_overlap_rate=0.01 if index % 2 == 0 else 0.99,
            )
            if campaign.split == "evaluation"
            else campaign
            for index, campaign in enumerate(campaigns)
        ]

        self.assertEqual(fit_models(modified), expected)

    def test_rejects_negative_campaign_overlap_standard_deviation(self) -> None:
        spec = load_experiment(SPEC)
        spec.populations[0].campaign_overlap_standard_deviation = -0.01

        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            validate_experiment(spec)

    def test_rejects_invalid_fingerprint_observation_parameter(self) -> None:
        spec = load_experiment(SPEC)
        spec.fingerprint_observation_model.conditional_match_probability = 0.0

        with self.assertRaisesRegex(ValueError, "conditional match probability"):
            validate_experiment(spec)

    def test_rejects_overlap_below_publisher_reach_feasibility_bound(self) -> None:
        spec = load_experiment(SPEC)
        population = spec.populations[0]
        for event_group in population.event_groups:
            event_group.reach_probability = 0.8
        population.overlap_rate_of_smaller_reach = 0.5

        with self.assertRaisesRegex(ValueError, "overlap rate must be at least"):
            validate_experiment(spec)

    def test_sampled_overlap_is_clipped_to_publisher_reach_feasibility_bound(self) -> None:
        spec = load_experiment(SPEC)
        population = spec.populations[0]
        for event_group in population.event_groups:
            event_group.reach_probability = 0.8
        population.overlap_rate_of_smaller_reach = 0.8
        generator = FixedGenerator(
            multinomial_results=[
                [1, 1, 1, population.population_size - 3],
                [1, 0, 0, 0, 0],
            ],
            fingerprint_counts=[1, 1],
            normal_value=-1.0,
        )

        _simulate_campaign(
            generator,
            "bounded-overlap",
            "reach",
            population,
            spec.fingerprint_observation_model,
            "train",
        )

        exposure_probabilities = generator.observed_probabilities[0]
        self.assertTrue(all(value >= 0.0 for value in exposure_probabilities))
        self.assertAlmostEqual(sum(exposure_probabilities), 1.0)
        self.assertAlmostEqual(exposure_probabilities[0], 0.6)
        self.assertAlmostEqual(exposure_probabilities[3], 0.0)
        fingerprint_probabilities = generator.observed_probabilities[1]
        self.assertAlmostEqual(fingerprint_probabilities[0], 0.144)
        self.assertAlmostEqual(fingerprint_probabilities[1], 0.096)
        self.assertAlmostEqual(fingerprint_probabilities[2], 0.06)
        self.assertAlmostEqual(fingerprint_probabilities[3], 0.56)
        self.assertAlmostEqual(fingerprint_probabilities[4], 0.14)

    def test_zero_realized_publisher_reach_has_clear_error(self) -> None:
        spec = load_experiment(SPEC)
        population = spec.populations[0]
        generator = FixedGenerator(
            multinomial_results=[[0, 0, 1, population.population_size - 1]],
            fingerprint_counts=[],
        )

        with self.assertRaisesRegex(ValueError, "zero realized reach"):
            _simulate_campaign(
                generator,
                "zero-reach",
                "reach",
                population,
                spec.fingerprint_observation_model,
                "train",
            )

    def test_zero_fingerprint_reach_has_clear_error(self) -> None:
        spec = load_experiment(SPEC)
        population = spec.populations[0]
        generator = FixedGenerator(
            multinomial_results=[
                [1, 1, 1, population.population_size - 3],
                [0, 0, 0, 0, 1],
            ],
            fingerprint_counts=[0, 0],
        )

        with self.assertRaisesRegex(ValueError, "zero fingerprint-bearing reach"):
            _simulate_campaign(
                generator,
                "zero-fingerprint",
                "reach",
                population,
                spec.fingerprint_observation_model,
                "train",
            )

    def test_single_publisher_reaches_are_invariant(self) -> None:
        campaigns = simulate_campaigns(load_experiment(SPEC))
        model = fit_models(campaigns)
        for campaign in campaigns:
            prediction = predict(campaign, model)
            fixed_overlap = (
                campaign.publisher_a_reach
                + campaign.publisher_b_reach
                - prediction.fixed_union
            )
            calibrated_overlap = (
                campaign.publisher_a_reach
                + campaign.publisher_b_reach
                - prediction.calibrated_union
            )
            self.assertGreaterEqual(fixed_overlap, 0.0)
            self.assertGreaterEqual(calibrated_overlap, 0.0)
            self.assertLessEqual(
                fixed_overlap,
                min(campaign.publisher_a_reach, campaign.publisher_b_reach),
            )
            self.assertLessEqual(
                calibrated_overlap, min(campaign.publisher_a_reach, campaign.publisher_b_reach)
            )

    def test_outputs_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            run_experiment(SPEC, first)
            run_experiment(SPEC, second)
            for filename in ["campaigns.csv", "predictions.csv", "model.textproto", "summary.json"]:
                self.assertEqual(
                    (Path(first) / filename).read_bytes(),
                    (Path(second) / filename).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
