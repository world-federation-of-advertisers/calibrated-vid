"""Synthetic campaign generation and fixed-versus-calibrated VID comparison."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from google.protobuf import text_format

from calibrated_vid import synthetic_campaign_experiment_pb2 as experiment_pb2


@dataclass(frozen=True)
class CampaignResult:
    campaign_id: str
    objective: str
    population: str
    split: str
    publisher_a_reach: int
    publisher_b_reach: int
    true_overlap: int
    true_union: int
    fingerprint_a_reach: int
    fingerprint_b_reach: int
    fingerprint_matches: int
    true_overlap_rate: float
    fingerprint_match_rate_from_a: float
    fingerprint_match_rate_from_b: float
    fingerprint_inferred_overlap_rate: float


@dataclass(frozen=True)
class Prediction:
    campaign_id: str
    objective: str
    split: str
    true_union: int
    fixed_union: float
    calibrated_union: float
    fixed_overlap_rate: float
    calibrated_overlap_rate: float
    fingerprint_inferred_overlap_rate: float
    fixed_union_error_percent: float
    calibrated_union_error_percent: float
    fixed_overlap_error_points: float
    calibrated_overlap_error_points: float


@dataclass(frozen=True)
class FittedModels:
    fixed_overlap_rate: float
    reference_overlap_center: float
    reference_sensitivity: float

    def as_proto(self) -> experiment_pb2.CalibratedVidModel:
        return experiment_pb2.CalibratedVidModel(
            base_overlap_rate=self.fixed_overlap_rate,
            reference_overlap_center=self.reference_overlap_center,
            reference_sensitivity=self.reference_sensitivity,
        )


def load_experiment(path: str | Path) -> experiment_pb2.SyntheticCampaignExperiment:
    spec = experiment_pb2.SyntheticCampaignExperiment()
    text_format.Parse(Path(path).read_text(encoding="utf-8"), spec)
    validate_experiment(spec)
    return spec


def validate_experiment(spec: experiment_pb2.SyntheticCampaignExperiment) -> None:
    fingerprint_model = spec.fingerprint_observation_model
    fingerprint_parameters = {
        "publisher A coverage": fingerprint_model.publisher_a_coverage,
        "publisher B coverage": fingerprint_model.publisher_b_coverage,
        "conditional match probability": fingerprint_model.conditional_match_probability,
    }
    for name, value in fingerprint_parameters.items():
        if not 0.0 < value <= 1.0:
            raise ValueError(f"Fingerprint {name} must be greater than zero and at most one")

    populations = {population.name: population for population in spec.populations}
    if not populations:
        raise ValueError("At least one virtual population is required")
    for population in populations.values():
        if population.population_size <= 0:
            raise ValueError(f"Population {population.name!r} has no members")
        if len(population.event_groups) != 2:
            raise ValueError(f"Population {population.name!r} must have two event groups")
        publishers = {event_group.publisher for event_group in population.event_groups}
        if publishers != {"publisher_a", "publisher_b"}:
            raise ValueError(
                f"Population {population.name!r} must define publisher_a and publisher_b"
            )
        rates = [event_group.reach_probability for event_group in population.event_groups]
        if not all(0.0 < rate < 1.0 for rate in rates):
            raise ValueError(f"Population {population.name!r} has an invalid reach probability")
        if not 0.0 < population.overlap_rate_of_smaller_reach < 1.0:
            raise ValueError(f"Population {population.name!r} has an invalid overlap rate")
        minimum_overlap_rate, _ = _feasible_overlap_rate_bounds(*rates)
        if population.overlap_rate_of_smaller_reach < minimum_overlap_rate:
            raise ValueError(
                f"Population {population.name!r} overlap rate must be at least "
                f"{minimum_overlap_rate:.6f} for its publisher reach probabilities"
            )
        if (
            not np.isfinite(population.campaign_overlap_standard_deviation)
            or population.campaign_overlap_standard_deviation < 0.0
        ):
            raise ValueError(
                f"Population {population.name!r} overlap standard deviation "
                "must be finite and nonnegative"
            )
    for cohort in spec.campaign_cohorts:
        if cohort.population not in populations:
            raise ValueError(f"Unknown population {cohort.population!r}")
        if cohort.campaign_count <= 0:
            raise ValueError(f"Cohort {cohort.objective!r} has no campaigns")
        if cohort.split == experiment_pb2.SyntheticCampaignExperiment.SPLIT_UNSPECIFIED:
            raise ValueError(f"Cohort {cohort.objective!r} has no split")


def _event_group_rates(population) -> tuple[float, float]:
    by_publisher = {
        event_group.publisher: event_group.reach_probability
        for event_group in population.event_groups
    }
    return by_publisher["publisher_a"], by_publisher["publisher_b"]


def _feasible_overlap_rate_bounds(
    reach_a_probability: float,
    reach_b_probability: float,
) -> tuple[float, float]:
    smaller_reach_probability = min(reach_a_probability, reach_b_probability)
    minimum_joint_probability = max(
        0.0, reach_a_probability + reach_b_probability - 1.0
    )
    return minimum_joint_probability / smaller_reach_probability, 1.0


def _simulate_campaign(
    rng: np.random.Generator,
    campaign_id: str,
    objective: str,
    population,
    fingerprint_model,
    split: str,
) -> CampaignResult:
    reach_a_probability, reach_b_probability = _event_group_rates(population)
    minimum_overlap_rate, maximum_overlap_rate = _feasible_overlap_rate_bounds(
        reach_a_probability, reach_b_probability
    )
    true_overlap_rate = float(
        np.clip(
            rng.normal(
                population.overlap_rate_of_smaller_reach,
                population.campaign_overlap_standard_deviation,
            ),
            minimum_overlap_rate,
            maximum_overlap_rate,
        )
    )
    joint_probability = true_overlap_rate * min(reach_a_probability, reach_b_probability)
    cell_probabilities = np.array(
        [
            joint_probability,
            reach_a_probability - joint_probability,
            reach_b_probability - joint_probability,
            1.0 - reach_a_probability - reach_b_probability + joint_probability,
        ]
    )
    if np.any(cell_probabilities < -1e-12):
        raise ValueError(f"Invalid four-cell probabilities for {campaign_id}: {cell_probabilities}")
    if np.any(cell_probabilities < 0.0):
        cell_probabilities = np.maximum(cell_probabilities, 0.0)
        cell_probabilities /= cell_probabilities.sum()
    both, a_only, b_only, _ = rng.multinomial(
        population.population_size, cell_probabilities
    )

    reach_a = int(both + a_only)
    reach_b = int(both + b_only)
    smaller_reach = min(reach_a, reach_b)
    if smaller_reach == 0:
        raise ValueError(
            f"Campaign {campaign_id!r} has zero realized reach on at least one publisher"
        )

    coverage_a = fingerprint_model.publisher_a_coverage
    coverage_b = fingerprint_model.publisher_b_coverage
    match_probability = fingerprint_model.conditional_match_probability
    fingerprint_match_observation_probability = (
        coverage_a * coverage_b * match_probability
    )
    both_fingerprint_probabilities = np.array(
        [
            fingerprint_match_observation_probability,
            coverage_a * coverage_b * (1.0 - match_probability),
            coverage_a * (1.0 - coverage_b),
            (1.0 - coverage_a) * coverage_b,
            (1.0 - coverage_a) * (1.0 - coverage_b),
        ]
    )
    (
        fingerprint_matches,
        both_fingerprints_unmatched,
        a_fingerprint_only,
        b_fingerprint_only,
        _,
    ) = rng.multinomial(both, both_fingerprint_probabilities)
    fingerprint_a_only_reach = int(rng.binomial(a_only, coverage_a))
    fingerprint_b_only_reach = int(rng.binomial(b_only, coverage_b))

    fingerprint_a = int(
        fingerprint_matches
        + both_fingerprints_unmatched
        + a_fingerprint_only
        + fingerprint_a_only_reach
    )
    fingerprint_b = int(
        fingerprint_matches
        + both_fingerprints_unmatched
        + b_fingerprint_only
        + fingerprint_b_only_reach
    )
    smaller_fingerprint_reach = min(fingerprint_a, fingerprint_b)
    if smaller_fingerprint_reach == 0:
        raise ValueError(
            f"Campaign {campaign_id!r} has zero fingerprint-bearing reach on at least one publisher"
        )
    return CampaignResult(
        campaign_id=campaign_id,
        objective=objective,
        population=population.name,
        split=split,
        publisher_a_reach=reach_a,
        publisher_b_reach=reach_b,
        true_overlap=int(both),
        true_union=int(reach_a + reach_b - both),
        fingerprint_a_reach=fingerprint_a,
        fingerprint_b_reach=fingerprint_b,
        fingerprint_matches=int(fingerprint_matches),
        true_overlap_rate=both / smaller_reach,
        fingerprint_match_rate_from_a=fingerprint_matches / fingerprint_a,
        fingerprint_match_rate_from_b=fingerprint_matches / fingerprint_b,
        fingerprint_inferred_overlap_rate=fingerprint_matches
        / (fingerprint_match_observation_probability * smaller_reach),
    )


def simulate_campaigns(
    spec: experiment_pb2.SyntheticCampaignExperiment,
) -> list[CampaignResult]:
    populations = {population.name: population for population in spec.populations}
    rng = np.random.default_rng(spec.random_seed)
    campaigns: list[CampaignResult] = []
    cohort_ordinals: dict[tuple[str, str], int] = {}
    for cohort in spec.campaign_cohorts:
        split = experiment_pb2.SyntheticCampaignExperiment.Split.Name(cohort.split).lower()
        key = (cohort.objective, split)
        start = cohort_ordinals.get(key, 0)
        for index in range(start, start + cohort.campaign_count):
            campaigns.append(
                _simulate_campaign(
                    rng,
                    f"{cohort.objective}-{split}-{index + 1:02d}",
                    cohort.objective,
                    populations[cohort.population],
                    spec.fingerprint_observation_model,
                    split,
                )
            )
        cohort_ordinals[key] = start + cohort.campaign_count
    return campaigns


def fit_models(campaigns: Iterable[CampaignResult]) -> FittedModels:
    training = [campaign for campaign in campaigns if campaign.split == "train"]
    if len(training) < 2:
        raise ValueError("At least two training campaigns are required")
    true_rates = np.asarray([campaign.true_overlap_rate for campaign in training])
    reference_rates = np.asarray(
        [campaign.fingerprint_inferred_overlap_rate for campaign in training]
    )
    reach_baseline = float(true_rates.mean())
    reference_center = reach_baseline
    centered_reference = reference_rates - reach_baseline
    denominator = float(np.dot(centered_reference, centered_reference))
    if denominator == 0.0:
        raise ValueError("Training campaigns have no fingerprint-overlap variation")
    sensitivity = float(
        np.dot(centered_reference, true_rates - reach_baseline) / denominator
    )
    return FittedModels(
        fixed_overlap_rate=reach_baseline,
        reference_overlap_center=reference_center,
        reference_sensitivity=sensitivity,
    )


def predict(campaign: CampaignResult, model: FittedModels) -> Prediction:
    smaller_reach = min(campaign.publisher_a_reach, campaign.publisher_b_reach)
    fixed_overlap_rate = model.fixed_overlap_rate
    calibrated_overlap_rate = float(
        np.clip(
            model.fixed_overlap_rate
            + model.reference_sensitivity
            * (
                campaign.fingerprint_inferred_overlap_rate
                - model.reference_overlap_center
            ),
            0.0,
            1.0,
        )
    )
    fixed_union = (
        campaign.publisher_a_reach
        + campaign.publisher_b_reach
        - fixed_overlap_rate * smaller_reach
    )
    calibrated_union = (
        campaign.publisher_a_reach
        + campaign.publisher_b_reach
        - calibrated_overlap_rate * smaller_reach
    )
    return Prediction(
        campaign_id=campaign.campaign_id,
        objective=campaign.objective,
        split=campaign.split,
        true_union=campaign.true_union,
        fixed_union=fixed_union,
        calibrated_union=calibrated_union,
        fixed_overlap_rate=fixed_overlap_rate,
        calibrated_overlap_rate=calibrated_overlap_rate,
        fingerprint_inferred_overlap_rate=campaign.fingerprint_inferred_overlap_rate,
        fixed_union_error_percent=100.0 * (fixed_union - campaign.true_union) / campaign.true_union,
        calibrated_union_error_percent=100.0
        * (calibrated_union - campaign.true_union)
        / campaign.true_union,
        fixed_overlap_error_points=100.0
        * (fixed_overlap_rate - campaign.true_overlap_rate),
        calibrated_overlap_error_points=100.0
        * (calibrated_overlap_rate - campaign.true_overlap_rate),
    )


def summarize(predictions: Iterable[Prediction]) -> list[dict[str, float | str | int]]:
    rows = list(predictions)
    summary: list[dict[str, float | str | int]] = []
    for objective in sorted({row.objective for row in rows}):
        cohort = [row for row in rows if row.objective == objective and row.split == "evaluation"]
        if not cohort:
            continue
        summary.append(
            {
                "objective": objective,
                "campaigns": len(cohort),
                "fixed_overlap_mae_points": float(
                    np.mean([abs(row.fixed_overlap_error_points) for row in cohort])
                ),
                "calibrated_overlap_mae_points": float(
                    np.mean([abs(row.calibrated_overlap_error_points) for row in cohort])
                ),
                "fixed_union_mape_percent": float(
                    np.mean([abs(row.fixed_union_error_percent) for row in cohort])
                ),
                "calibrated_union_mape_percent": float(
                    np.mean([abs(row.calibrated_union_error_percent) for row in cohort])
                ),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    spec_path: str | Path,
    output_dir: str | Path,
) -> tuple[list[CampaignResult], list[Prediction], FittedModels, list[dict]]:
    spec = load_experiment(spec_path)
    campaigns = simulate_campaigns(spec)
    model = fit_models(campaigns)
    predictions = [predict(campaign, model) for campaign in campaigns]
    summary = summarize(predictions)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "campaigns.csv", [asdict(campaign) for campaign in campaigns])
    _write_csv(output / "predictions.csv", [asdict(prediction) for prediction in predictions])
    (output / "model.textproto").write_text(
        text_format.MessageToString(model.as_proto()), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "model": asdict(model),
                "evaluation": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return campaigns, predictions, model, summary
