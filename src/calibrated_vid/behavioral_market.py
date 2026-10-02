"""Behavioral synthetic market for campaign-overlap calibration experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SegmentSpec:
    """One latent audience segment used to generate publisher opportunity."""

    name: str
    share: float
    feed_activity_a: float
    feed_activity_b: float
    short_video_activity_a: float
    short_video_activity_b: float
    click_score_a: float
    click_score_b: float
    conversion_score: float
    cost_a: float
    cost_b: float
    fingerprint_multiplier: float


@dataclass(frozen=True)
class MarketConfig:
    population_size: int = 240_000
    medium_reach: int = 12_000
    large_reach: int = 48_000
    fingerprint_coverage_a: float = 0.30
    fingerprint_coverage_b: float = 0.80
    fingerprint_agreement: float = 0.60
    opportunity_shift_sigma: float = 0.70
    price_shift_sigma: float = 0.60
    direction_cost_contrast: float = 1.40
    traffic_response_exponent: float = 2.00
    conversion_response_exponent: float = 2.50
    selection_concentration: float = 2.00
    objective_comparisons: int = 48
    traffic_comparisons: int = 32
    conversion_comparisons: int = 20
    direction_comparisons: int = 32
    random_seed: int = 20260927


@dataclass(frozen=True)
class CampaignPlan:
    campaign_id: str
    comparison_id: str
    scenario: str
    objective: str
    split: str
    reach_a: int
    reach_b: int
    active_days: int
    short_video_share_a: float
    short_video_share_b: float
    auction_state: int
    opportunity_asymmetry: float


@dataclass(frozen=True)
class CampaignResult:
    campaign_id: str
    comparison_id: str
    scenario: str
    objective: str
    split: str
    direction: str
    reach_a: int
    reach_b: int
    active_days: int
    short_video_share_a: float
    short_video_share_b: float
    auction_state: int
    opportunity_asymmetry: float
    overlap: int
    union_reach: int
    overlap_rate: float
    fingerprint_a_reach: int
    fingerprint_b_reach: int
    fingerprint_matches: int
    fingerprint_overlap_estimate: float


@dataclass(frozen=True)
class FittedOverlapModel:
    fixed_overlap_rate: float
    reference_center: float
    reference_sensitivity: float


@dataclass(frozen=True)
class CampaignPrediction:
    campaign_id: str
    scenario: str
    objective: str
    split: str
    direction: str
    true_overlap_rate: float
    reference_overlap_rate: float
    fixed_overlap_rate: float
    calibrated_overlap_rate: float
    true_union: int
    fixed_union: float
    calibrated_union: float
    true_incremental_unique_reach: int
    fixed_incremental_unique_reach: float
    calibrated_incremental_unique_reach: float
    fixed_overlap_error_points: float
    calibrated_overlap_error_points: float
    fixed_union_error_percent: float
    calibrated_union_error_percent: float
    fixed_incremental_error_percent: float
    calibrated_incremental_error_percent: float


@dataclass(frozen=True)
class SyntheticAudience:
    segment: np.ndarray
    feed_activity_a: np.ndarray
    feed_activity_b: np.ndarray
    short_video_activity_a: np.ndarray
    short_video_activity_b: np.ndarray
    click_score_a: np.ndarray
    click_score_b: np.ndarray
    conversion_score: np.ndarray
    cost_a: np.ndarray
    cost_b: np.ndarray
    fingerprint_a: np.ndarray
    fingerprint_b: np.ndarray
    fingerprint_agreement: np.ndarray


SEGMENTS: tuple[SegmentSpec, ...] = (
    SegmentSpec("dual_feed_regulars", 0.22, 0.92, 0.88, 0.42, 0.48, 0.018, 0.018, 0.003, 0.35, 0.35, 1.15),
    SegmentSpec("publisher_a_regulars", 0.15, 0.95, 0.24, 0.72, 0.11, 0.016, 0.006, 0.002, 0.85, 1.24, 0.85),
    SegmentSpec("publisher_b_regulars", 0.18, 0.20, 0.96, 0.10, 0.79, 0.006, 0.016, 0.002, 1.34, 0.85, 0.85),
    SegmentSpec("publisher_a_clickers", 0.12, 0.86, 0.32, 0.66, 0.20, 0.45, 0.002, 0.018, 1.28, 1.02, 0.95),
    SegmentSpec("publisher_b_clickers", 0.12, 0.31, 0.87, 0.18, 0.72, 0.002, 0.45, 0.018, 1.02, 1.31, 0.95),
    SegmentSpec("short_video_a_regulars", 0.08, 0.25, 0.16, 0.96, 0.23, 0.080, 0.005, 0.008, 0.70, 1.10, 0.90),
    SegmentSpec("short_video_b_regulars", 0.08, 0.17, 0.24, 0.20, 0.97, 0.005, 0.080, 0.008, 1.12, 0.58, 0.90),
    SegmentSpec("high_intent_dual", 0.03, 0.67, 0.66, 0.48, 0.51, 0.18, 0.18, 0.060, 2.10, 2.10, 1.20),
    SegmentSpec("occasional_dual", 0.02, 0.32, 0.31, 0.20, 0.19, 0.025, 0.025, 0.006, 0.84, 0.84, 0.90),
)


def _segment_array(field: str, assignments: np.ndarray) -> np.ndarray:
    values = np.asarray([getattr(segment, field) for segment in SEGMENTS], dtype=float)
    return values[assignments]


def build_audience(config: MarketConfig = MarketConfig()) -> SyntheticAudience:
    """Creates one fixed synthetic population shared by every campaign."""

    rng = np.random.default_rng(config.random_seed)
    shares = np.asarray([segment.share for segment in SEGMENTS], dtype=float)
    if not np.isclose(shares.sum(), 1.0):
        raise ValueError("Segment shares must sum to one")
    segment = rng.choice(len(SEGMENTS), size=config.population_size, p=shares)

    common_activity_noise = rng.lognormal(mean=0.0, sigma=0.20, size=config.population_size)
    publisher_a_noise = rng.lognormal(mean=0.0, sigma=0.16, size=config.population_size)
    publisher_b_noise = rng.lognormal(mean=0.0, sigma=0.16, size=config.population_size)
    response_noise_a = rng.lognormal(mean=0.0, sigma=0.28, size=config.population_size)
    response_noise_b = rng.lognormal(mean=0.0, sigma=0.28, size=config.population_size)
    conversion_noise = rng.lognormal(mean=0.0, sigma=0.28, size=config.population_size)
    cost_noise = rng.lognormal(mean=0.0, sigma=0.16, size=config.population_size)

    feed_activity_a = np.clip(
        _segment_array("feed_activity_a", segment)
        * common_activity_noise
        * publisher_a_noise,
        0.01,
        1.0,
    )
    feed_activity_b = np.clip(
        _segment_array("feed_activity_b", segment)
        * common_activity_noise
        * publisher_b_noise,
        0.01,
        1.0,
    )
    short_video_activity_a = np.clip(
        _segment_array("short_video_activity_a", segment)
        * common_activity_noise
        * publisher_a_noise,
        0.01,
        1.0,
    )
    short_video_activity_b = np.clip(
        _segment_array("short_video_activity_b", segment)
        * common_activity_noise
        * publisher_b_noise,
        0.01,
        1.0,
    )
    click_score_a = np.clip(
        _segment_array("click_score_a", segment) * response_noise_a,
        1e-4,
        1.0,
    )
    click_score_b = np.clip(
        _segment_array("click_score_b", segment) * response_noise_b,
        1e-4,
        1.0,
    )
    conversion_score = np.clip(
        _segment_array("conversion_score", segment) * conversion_noise,
        1e-5,
        1.0,
    )
    cost_a = _segment_array("cost_a", segment) * cost_noise
    cost_b = _segment_array("cost_b", segment) * cost_noise
    fingerprint_multiplier = _segment_array("fingerprint_multiplier", segment)
    fingerprint_a = rng.random(config.population_size) < np.clip(
        config.fingerprint_coverage_a * fingerprint_multiplier, 0.0, 1.0
    )
    fingerprint_b = rng.random(config.population_size) < np.clip(
        config.fingerprint_coverage_b * fingerprint_multiplier, 0.0, 1.0
    )
    fingerprint_agreement = (
        fingerprint_a
        & fingerprint_b
        & (rng.random(config.population_size) < config.fingerprint_agreement)
    )
    return SyntheticAudience(
        segment=segment,
        feed_activity_a=feed_activity_a,
        feed_activity_b=feed_activity_b,
        short_video_activity_a=short_video_activity_a,
        short_video_activity_b=short_video_activity_b,
        click_score_a=click_score_a,
        click_score_b=click_score_b,
        conversion_score=conversion_score,
        cost_a=cost_a,
        cost_b=cost_b,
        fingerprint_a=fingerprint_a,
        fingerprint_b=fingerprint_b,
        fingerprint_agreement=fingerprint_agreement,
    )


def _direction(reach_a: int, reach_b: int) -> str:
    if reach_a == reach_b:
        return "large_large"
    return "medium_large" if reach_a < reach_b else "large_medium"


def make_campaign_plans(config: MarketConfig = MarketConfig()) -> list[CampaignPlan]:
    """Builds balanced training and evaluation cohorts for three stress tests."""

    if not 0 <= config.conversion_comparisons <= config.traffic_comparisons:
        raise ValueError("conversion_comparisons must not exceed traffic_comparisons")
    if not 0 <= config.traffic_comparisons <= config.objective_comparisons:
        raise ValueError("traffic_comparisons must not exceed objective_comparisons")
    rng = np.random.default_rng(config.random_seed + 1)
    plans: list[CampaignPlan] = []
    for split in ("train", "evaluation"):
        for ordinal in range(config.objective_comparisons):
            comparison_id = f"objective-{split}-{ordinal + 1:02d}"
            active_days = int(
                np.clip(
                    round(
                        5.0
                        + 7.0 * rng.lognormal(np.log(2.6), 0.48)
                        + rng.normal(0.0, 5.0)
                    ),
                    7,
                    60,
                )
            )
            base_short_video = float(rng.beta(1.4, 2.1))
            placement_asymmetry = float(rng.normal(0.0, 0.24))
            short_video_share_a = float(
                np.clip(base_short_video + placement_asymmetry, 0.0, 1.0)
            )
            short_video_share_b = float(
                np.clip(base_short_video - placement_asymmetry, 0.0, 1.0)
            )
            auction_state = int(rng.integers(0, 2**31 - 1))
            objectives = [("reach", "reach_large_large")]
            if ordinal < config.traffic_comparisons:
                objectives.append(("traffic", "traffic_large_large"))
            if ordinal < config.conversion_comparisons:
                objectives.append(("conversion", "conversion_large_large"))
            for objective, scenario in objectives:
                plans.append(
                    CampaignPlan(
                        campaign_id=f"{scenario}-{split}-{ordinal + 1:02d}",
                        comparison_id=comparison_id,
                        scenario=scenario,
                        objective=objective,
                        split=split,
                        reach_a=config.large_reach,
                        reach_b=config.large_reach,
                        active_days=active_days,
                        short_video_share_a=short_video_share_a,
                        short_video_share_b=short_video_share_b,
                        auction_state=auction_state,
                        opportunity_asymmetry=0.0,
                    )
                )

        for ordinal in range(config.direction_comparisons):
            comparison_id = f"direction-{split}-{ordinal + 1:02d}"
            active_days = int(
                np.clip(
                    round(
                        7.0
                        + 7.0 * rng.lognormal(np.log(2.7), 0.28)
                        + rng.normal(0.0, 3.0)
                    ),
                    10,
                    50,
                )
            )
            _base_short_video = float(rng.beta(1.5, 2.0))
            placement_asymmetry = float(rng.normal(0.0, 0.20))
            short_video_share_a = float(
                np.clip(0.12 + 0.04 * placement_asymmetry, 0.0, 0.30)
            )
            short_video_share_b = float(
                np.clip(0.82 - 0.04 * placement_asymmetry, 0.60, 1.0)
            )
            auction_state = int(rng.integers(0, 2**31 - 1))
            for scenario, reach_a, reach_b in (
                ("reach_medium_large", config.medium_reach, config.large_reach),
                ("reach_large_medium", config.large_reach, config.medium_reach),
            ):
                plans.append(
                    CampaignPlan(
                        campaign_id=f"{scenario}-{split}-{ordinal + 1:02d}",
                        comparison_id=comparison_id,
                        scenario=scenario,
                        objective="reach",
                        split=split,
                        reach_a=reach_a,
                        reach_b=reach_b,
                        active_days=active_days,
                        short_video_share_a=short_video_share_a,
                        short_video_share_b=short_video_share_b,
                        auction_state=auction_state,
                        opportunity_asymmetry=config.direction_cost_contrast,
                    )
                )
    return plans


def _delivery_weight(
    audience: SyntheticAudience,
    plan: CampaignPlan,
    publisher: str,
    config: MarketConfig,
) -> np.ndarray:
    if publisher == "a":
        activity = (
            (1.0 - plan.short_video_share_a) * audience.feed_activity_a
            + plan.short_video_share_a * audience.short_video_activity_a
        )
        click_score = audience.click_score_a
        cost = audience.cost_a
        publisher_offset = 17
    elif publisher == "b":
        activity = (
            (1.0 - plan.short_video_share_b) * audience.feed_activity_b
            + plan.short_video_share_b * audience.short_video_activity_b
        )
        click_score = audience.click_score_b
        cost = audience.cost_b
        publisher_offset = 31
    else:
        raise ValueError(f"Unknown publisher {publisher!r}")

    if plan.objective == "reach":
        objective_value = np.ones_like(cost)
    elif plan.objective == "traffic":
        objective_value = np.power(click_score, config.traffic_response_exponent)
    elif plan.objective == "conversion":
        objective_value = np.power(
            audience.conversion_score,
            config.conversion_response_exponent,
        )
    else:
        raise ValueError(f"Unknown objective {plan.objective!r}")

    publisher_rng = np.random.default_rng(plan.auction_state + publisher_offset)
    local_opportunity_shift = publisher_rng.normal(
        0.0,
        config.opportunity_shift_sigma,
        len(SEGMENTS),
    )
    local_price_shift = publisher_rng.normal(
        0.0,
        config.price_shift_sigma,
        len(SEGMENTS),
    )
    local_activity = activity * np.exp(local_opportunity_shift[audience.segment])
    opportunity_probability = 1.0 - np.exp(
        -0.08 * plan.active_days * local_activity
    )
    campaign_cost = cost * np.exp(local_price_shift[audience.segment])
    if plan.opportunity_asymmetry:
        dual_regulars = audience.segment == 0
        publisher_b_heavy = audience.segment == 6
        other_publisher_b_regulars = np.isin(audience.segment, (2, 4))
        if publisher == "a":
            campaign_cost = campaign_cost * np.where(
                dual_regulars,
                np.exp(-plan.opportunity_asymmetry),
                np.where(
                    publisher_b_heavy,
                    np.exp(0.50 * plan.opportunity_asymmetry),
                    1.0,
                ),
            )
        else:
            campaign_cost = campaign_cost * np.where(
                publisher_b_heavy,
                np.exp(-plan.opportunity_asymmetry),
                np.where(
                    dual_regulars,
                    1.0,
                    np.where(
                        other_publisher_b_regulars,
                        np.exp(0.50 * plan.opportunity_asymmetry),
                        1.0,
                    ),
                ),
            )

    return (
        np.maximum(opportunity_probability, 1e-8)
        * np.maximum(objective_value / campaign_cost, 1e-8)
    )


def _weighted_sample(
    rng: np.random.Generator,
    weights: np.ndarray,
    count: int,
    concentration: float,
) -> np.ndarray:
    if not 0 < count < len(weights):
        raise ValueError("Reach count must be positive and smaller than the population")
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("Delivery weights must be finite and positive")
    gumbel = rng.gumbel(size=len(weights))
    keys = concentration * np.log(weights) + gumbel
    return np.argpartition(keys, -count)[-count:]


def _sampling_seed(random_seed: int, comparison_id: str) -> np.random.SeedSequence:
    """Returns one stable seed per matched campaign comparison."""

    return np.random.SeedSequence(
        [random_seed, *comparison_id.encode("utf-8")]
    )


def simulate_campaign(
    audience: SyntheticAudience,
    plan: CampaignPlan,
    config: MarketConfig = MarketConfig(),
) -> CampaignResult:
    """Simulates publisher delivery and derives overlap from selected people."""

    rng = np.random.default_rng(
        _sampling_seed(config.random_seed, plan.comparison_id)
    )
    concentration = config.selection_concentration
    selected_a = _weighted_sample(
        rng,
        _delivery_weight(audience, plan, "a", config),
        plan.reach_a,
        concentration,
    )
    selected_b = _weighted_sample(
        rng,
        _delivery_weight(audience, plan, "b", config),
        plan.reach_b,
        concentration,
    )
    reached_a = np.zeros(config.population_size, dtype=bool)
    reached_b = np.zeros(config.population_size, dtype=bool)
    reached_a[selected_a] = True
    reached_b[selected_b] = True
    both = reached_a & reached_b
    overlap = int(both.sum())
    smaller_reach = min(plan.reach_a, plan.reach_b)
    fingerprint_a_reach = int((reached_a & audience.fingerprint_a).sum())
    fingerprint_b_reach = int((reached_b & audience.fingerprint_b).sum())
    fingerprint_matches = int((both & audience.fingerprint_agreement).sum())
    realized_coverage_a = fingerprint_a_reach / plan.reach_a
    realized_coverage_b = fingerprint_b_reach / plan.reach_b
    observation_probability = (
        realized_coverage_a * realized_coverage_b * config.fingerprint_agreement
    )
    if observation_probability <= 0.0:
        raise ValueError(f"Campaign {plan.campaign_id} has no usable fingerprint coverage")
    fingerprint_overlap_estimate = min(
        1.0,
        fingerprint_matches / (observation_probability * smaller_reach),
    )
    return CampaignResult(
        campaign_id=plan.campaign_id,
        comparison_id=plan.comparison_id,
        scenario=plan.scenario,
        objective=plan.objective,
        split=plan.split,
        direction=_direction(plan.reach_a, plan.reach_b),
        reach_a=plan.reach_a,
        reach_b=plan.reach_b,
        active_days=plan.active_days,
        short_video_share_a=plan.short_video_share_a,
        short_video_share_b=plan.short_video_share_b,
        auction_state=plan.auction_state,
        opportunity_asymmetry=plan.opportunity_asymmetry,
        overlap=overlap,
        union_reach=plan.reach_a + plan.reach_b - overlap,
        overlap_rate=overlap / smaller_reach,
        fingerprint_a_reach=fingerprint_a_reach,
        fingerprint_b_reach=fingerprint_b_reach,
        fingerprint_matches=fingerprint_matches,
        fingerprint_overlap_estimate=fingerprint_overlap_estimate,
    )


def simulate_market(config: MarketConfig = MarketConfig()) -> list[CampaignResult]:
    audience = build_audience(config)
    return [simulate_campaign(audience, plan, config) for plan in make_campaign_plans(config)]


def fit_overlap_model(campaigns: Iterable[CampaignResult]) -> FittedOverlapModel:
    training = [
        campaign
        for campaign in campaigns
        if campaign.split == "train" and campaign.scenario == "reach_large_large"
    ]
    if len(training) < 2:
        raise ValueError("At least two large Reach training campaigns are required")
    truth = np.asarray([campaign.overlap_rate for campaign in training])
    reference = np.asarray(
        [campaign.fingerprint_overlap_estimate for campaign in training]
    )
    baseline = float(np.mean(truth))
    reference_center = float(np.mean(reference))
    centered = reference - reference_center
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        raise ValueError("Training campaigns have no reference variation")
    sensitivity = float(np.dot(centered, truth - baseline) / denominator)
    return FittedOverlapModel(baseline, reference_center, sensitivity)


def predict_campaign(
    campaign: CampaignResult,
    model: FittedOverlapModel,
) -> CampaignPrediction:
    smaller_reach = min(campaign.reach_a, campaign.reach_b)
    calibrated_rate = float(
        np.clip(
            model.fixed_overlap_rate
            + model.reference_sensitivity
            * (campaign.fingerprint_overlap_estimate - model.reference_center),
            0.0,
            1.0,
        )
    )
    fixed_union = campaign.reach_a + campaign.reach_b - model.fixed_overlap_rate * smaller_reach
    calibrated_union = campaign.reach_a + campaign.reach_b - calibrated_rate * smaller_reach
    true_incremental = smaller_reach - campaign.overlap
    fixed_incremental = smaller_reach * (1.0 - model.fixed_overlap_rate)
    calibrated_incremental = smaller_reach * (1.0 - calibrated_rate)
    if true_incremental <= 0:
        raise ValueError(f"Campaign {campaign.campaign_id} has no incremental unique reach")
    return CampaignPrediction(
        campaign_id=campaign.campaign_id,
        scenario=campaign.scenario,
        objective=campaign.objective,
        split=campaign.split,
        direction=campaign.direction,
        true_overlap_rate=campaign.overlap_rate,
        reference_overlap_rate=campaign.fingerprint_overlap_estimate,
        fixed_overlap_rate=model.fixed_overlap_rate,
        calibrated_overlap_rate=calibrated_rate,
        true_union=campaign.union_reach,
        fixed_union=fixed_union,
        calibrated_union=calibrated_union,
        true_incremental_unique_reach=true_incremental,
        fixed_incremental_unique_reach=fixed_incremental,
        calibrated_incremental_unique_reach=calibrated_incremental,
        fixed_overlap_error_points=100.0 * (model.fixed_overlap_rate - campaign.overlap_rate),
        calibrated_overlap_error_points=100.0 * (calibrated_rate - campaign.overlap_rate),
        fixed_union_error_percent=100.0 * (fixed_union - campaign.union_reach) / campaign.union_reach,
        calibrated_union_error_percent=100.0
        * (calibrated_union - campaign.union_reach)
        / campaign.union_reach,
        fixed_incremental_error_percent=100.0
        * (fixed_incremental - true_incremental)
        / true_incremental,
        calibrated_incremental_error_percent=100.0
        * (calibrated_incremental - true_incremental)
        / true_incremental,
    )


def summarize_scenarios(
    campaigns: Iterable[CampaignResult],
    predictions: Iterable[CampaignPrediction],
) -> list[dict[str, float | int | str]]:
    campaign_rows = {campaign.campaign_id: campaign for campaign in campaigns}
    evaluation = [prediction for prediction in predictions if prediction.split == "evaluation"]
    rows: list[dict[str, float | int | str]] = []
    for scenario in sorted({prediction.scenario for prediction in evaluation}):
        cohort = [prediction for prediction in evaluation if prediction.scenario == scenario]
        overlap = np.asarray([prediction.true_overlap_rate for prediction in cohort])
        fixed_error = np.abs(
            np.asarray([prediction.fixed_overlap_error_points for prediction in cohort])
        )
        calibrated_error = np.abs(
            np.asarray([prediction.calibrated_overlap_error_points for prediction in cohort])
        )
        union_error = np.abs(
            np.asarray([prediction.fixed_union_error_percent for prediction in cohort])
        )
        calibrated_union_error = np.abs(
            np.asarray([prediction.calibrated_union_error_percent for prediction in cohort])
        )
        fixed_incremental_error = np.abs(
            np.asarray([prediction.fixed_incremental_error_percent for prediction in cohort])
        )
        calibrated_incremental_error = np.abs(
            np.asarray(
                [prediction.calibrated_incremental_error_percent for prediction in cohort]
            )
        )
        calibrated_overlap = np.asarray(
            [prediction.calibrated_overlap_rate for prediction in cohort]
        )
        sample = campaign_rows[cohort[0].campaign_id]
        rows.append(
            {
                "scenario": scenario,
                "objective": sample.objective,
                "direction": sample.direction,
                "campaigns": len(cohort),
                "overlap_p10_percent": float(100.0 * np.quantile(overlap, 0.10)),
                "overlap_median_percent": float(100.0 * np.median(overlap)),
                "overlap_p90_percent": float(100.0 * np.quantile(overlap, 0.90)),
                "fixed_overlap_percent": float(
                    100.0 * cohort[0].fixed_overlap_rate
                ),
                "calibrated_overlap_median_percent": float(
                    100.0 * np.median(calibrated_overlap)
                ),
                "fixed_overlap_mae_points": float(np.mean(fixed_error)),
                "fixed_overlap_p90_error_points": float(np.quantile(fixed_error, 0.90)),
                "calibrated_overlap_mae_points": float(np.mean(calibrated_error)),
                "calibrated_overlap_p90_error_points": float(
                    np.quantile(calibrated_error, 0.90)
                ),
                "fixed_union_mape_percent": float(np.mean(union_error)),
                "calibrated_union_mape_percent": float(
                    np.mean(calibrated_union_error)
                ),
                "fixed_incremental_mape_percent": float(
                    np.mean(fixed_incremental_error)
                ),
                "calibrated_incremental_mape_percent": float(
                    np.mean(calibrated_incremental_error)
                ),
                "fixed_incremental_p90_error_percent": float(
                    np.quantile(fixed_incremental_error, 0.90)
                ),
                "calibrated_incremental_p90_error_percent": float(
                    np.quantile(calibrated_incremental_error, 0.90)
                ),
            }
        )
    return rows


def behavioral_diagnostics(
    campaigns: Iterable[CampaignResult],
) -> dict[str, dict[str, float | int]]:
    evaluation = [campaign for campaign in campaigns if campaign.split == "evaluation"]
    by_comparison: dict[str, dict[str, CampaignResult]] = {}
    for campaign in evaluation:
        by_comparison.setdefault(campaign.comparison_id, {})[campaign.scenario] = campaign

    objective_differences = np.asarray(
        [
            rows["traffic_large_large"].overlap_rate
            - rows["reach_large_large"].overlap_rate
            for rows in by_comparison.values()
            if "traffic_large_large" in rows and "reach_large_large" in rows
        ]
    )
    direction_differences = np.asarray(
        [
            rows["reach_medium_large"].overlap_rate
            - rows["reach_large_medium"].overlap_rate
            for rows in by_comparison.values()
            if "reach_medium_large" in rows and "reach_large_medium" in rows
        ]
    )
    reach_large = [
        campaign for campaign in evaluation if campaign.scenario == "reach_large_large"
    ]
    overlap = np.asarray([campaign.overlap_rate for campaign in reach_large])
    p10 = float(100.0 * np.quantile(overlap, 0.10))
    p90 = float(100.0 * np.quantile(overlap, 0.90))
    return {
        "paired_objective_test": {
            "pairs": len(objective_differences),
            "traffic_lower_share": float(np.mean(objective_differences < 0.0)),
            "traffic_minus_reach_median_points": float(
                100.0 * np.median(objective_differences)
            ),
            "traffic_minus_reach_p10_points": float(
                100.0 * np.quantile(objective_differences, 0.10)
            ),
            "traffic_minus_reach_p90_points": float(
                100.0 * np.quantile(objective_differences, 0.90)
            ),
        },
        "paired_direction_test": {
            "pairs": len(direction_differences),
            "medium_large_higher_share": float(np.mean(direction_differences > 0.0)),
            "medium_large_minus_large_medium_median_points": float(
                100.0 * np.median(direction_differences)
            ),
            "medium_large_minus_large_medium_p10_points": float(
                100.0 * np.quantile(direction_differences, 0.10)
            ),
            "medium_large_minus_large_medium_p90_points": float(
                100.0 * np.quantile(direction_differences, 0.90)
            ),
        },
        "within_large_reach": {
            "campaigns": len(reach_large),
            "overlap_p10_percent": p10,
            "overlap_median_percent": float(100.0 * np.median(overlap)),
            "overlap_p90_percent": p90,
            "overlap_p10_p90_width_points": p90 - p10,
        },
    }


def segment_table() -> list[dict[str, float | str]]:
    return [asdict(segment) for segment in SEGMENTS]


def run_behavioral_experiment(
    output_directory: str | Path,
    config: MarketConfig = MarketConfig(),
) -> tuple[
    list[CampaignResult],
    list[CampaignPrediction],
    FittedOverlapModel,
    dict[str, object],
]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    campaigns = simulate_market(config)
    model = fit_overlap_model(campaigns)
    predictions = [predict_campaign(campaign, model) for campaign in campaigns]
    report: dict[str, object] = {
        "config": asdict(config),
        "model": asdict(model),
        "scenarios": summarize_scenarios(campaigns, predictions),
        "diagnostics": behavioral_diagnostics(campaigns),
    }

    _write_csv(output / "segments.csv", segment_table())
    _write_csv(output / "campaigns.csv", [asdict(campaign) for campaign in campaigns])
    _write_csv(output / "predictions.csv", [asdict(prediction) for prediction in predictions])
    (output / "summary.json").write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return campaigns, predictions, model, report


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
