"""Interpretable person-level market for overlap-transfer experiments.

The module intentionally models people and publisher-local delivery, not RID-to-VID
assignment.  It asks whether one overlap learned from canonical large Reach
campaigns transfers to other objectives, publisher-size directions, and campaign
conditions.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SegmentSpec:
    """One interpretable activity and response pattern in the eligible market."""

    name: str
    share: float
    feed_a: float
    feed_b: float
    video_a: float
    video_b: float
    daytime_share: float
    click_a: float
    click_b: float
    cost_a: float
    cost_b: float


@dataclass(frozen=True)
class CampaignProfile:
    """Observable campaign conditions that change publisher-local availability."""

    name: str
    label: str
    active_days: int
    video_share_a: float
    video_share_b: float
    schedule: str


@dataclass(frozen=True)
class IntuitiveConfig:
    population_size: int = 180_000
    medium_reach: int = 10_800
    large_reach: int = 36_000
    activity_contrast: float = 1.00
    response_contrast: float = 1.0
    profile_strength: float = 1.0
    traffic_response_exponent: float = 0.50
    selection_concentration: float = 12.0
    training_replicates: int = 24
    objective_replicates_per_profile: int = 8
    direction_replicates: int = 24
    large_profile_replicates: int = 10
    reference_coverage_a: float = 0.30
    reference_coverage_b: float = 0.80
    reference_agreement: float = 0.60
    random_seed: int = 20261002


@dataclass(frozen=True)
class CampaignPlan:
    campaign_id: str
    comparison_id: str
    scenario: str
    split: str
    objective: str
    profile: str
    reach_a: int
    reach_b: int


@dataclass(frozen=True)
class CampaignResult:
    campaign_id: str
    comparison_id: str
    scenario: str
    split: str
    objective: str
    profile: str
    direction: str
    reach_a: int
    reach_b: int
    overlap: int
    overlap_rate: float
    union_reach: int
    reference_matches: int
    reference_observation_probability: float
    reference_overlap_estimate: float


@dataclass(frozen=True)
class SyntheticPopulation:
    segment: np.ndarray
    feed_a: np.ndarray
    feed_b: np.ndarray
    video_a: np.ndarray
    video_b: np.ndarray
    daytime_share: np.ndarray
    click_a: np.ndarray
    click_b: np.ndarray
    cost_a: np.ndarray
    cost_b: np.ndarray
    reference_a: np.ndarray
    reference_b: np.ndarray
    reference_agrees: np.ndarray


SEGMENTS: tuple[SegmentSpec, ...] = (
    # Publisher A's most active cohort contains many dual-publisher regulars.
    SegmentSpec("dual_daily", 0.24, 0.95, 0.86, 0.64, 0.61, 0.55, 0.025, 0.025, 0.72, 0.78),
    SegmentSpec("a_regular_dual_occasional", 0.17, 0.96, 0.48, 0.76, 0.30, 0.62, 0.022, 0.009, 0.83, 1.02),
    # Publisher B's most active cohort contains more publisher-heavy users.
    SegmentSpec("b_heavy_regular", 0.23, 0.14, 0.98, 0.10, 0.86, 0.42, 0.006, 0.021, 1.18, 0.76),
    SegmentSpec("dual_video_regular", 0.10, 0.38, 0.36, 0.96, 0.93, 0.38, 0.040, 0.041, 0.80, 0.80),
    SegmentSpec("a_response_regular", 0.09, 0.79, 0.34, 0.70, 0.28, 0.58, 0.36, 0.012, 1.04, 1.10),
    SegmentSpec("b_response_regular", 0.09, 0.31, 0.82, 0.24, 0.75, 0.44, 0.012, 0.36, 1.10, 1.04),
    SegmentSpec("daytime_dual", 0.04, 0.61, 0.58, 0.33, 0.34, 0.90, 0.035, 0.035, 0.91, 0.91),
    SegmentSpec("evening_video_dual", 0.02, 0.26, 0.25, 0.78, 0.76, 0.12, 0.060, 0.060, 0.86, 0.86),
    SegmentSpec("occasional_dual", 0.02, 0.22, 0.21, 0.18, 0.17, 0.52, 0.018, 0.018, 0.96, 0.96),
)


PROFILES: tuple[CampaignProfile, ...] = (
    CampaignProfile("broad_long", "Long flight, broad placements", 42, 0.35, 0.35, "broad"),
    CampaignProfile("feed_short", "Short flight, feed-heavy", 10, 0.05, 0.05, "broad"),
    CampaignProfile("video_short", "Short flight, video-heavy", 10, 0.90, 0.90, "broad"),
    CampaignProfile("a_feed_b_video", "A feed-heavy, B video-heavy", 18, 0.08, 0.88, "broad"),
    CampaignProfile("a_video_b_feed", "A video-heavy, B feed-heavy", 18, 0.88, 0.08, "broad"),
    CampaignProfile(
        "daytime_feed",
        "Both publishers: daytime-skewed, feed-heavy",
        18,
        0.10,
        0.10,
        "daytime",
    ),
    CampaignProfile(
        "evening_video",
        "Both publishers: evening-skewed, video-heavy",
        18,
        0.82,
        0.82,
        "evening",
    ),
)

PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}


def _values(field: str, segment: np.ndarray) -> np.ndarray:
    return np.asarray([getattr(row, field) for row in SEGMENTS], dtype=float)[segment]


def _blend_pair(a: np.ndarray, b: np.ndarray, contrast: float) -> tuple[np.ndarray, np.ndarray]:
    """Moves publisher-specific values toward their person-level mean at contrast=0."""

    midpoint = 0.5 * (a + b)
    return (
        np.clip(midpoint + contrast * (a - midpoint), 0.005, 1.0),
        np.clip(midpoint + contrast * (b - midpoint), 0.005, 1.0),
    )


def _blend_positive_pair(
    a: np.ndarray,
    b: np.ndarray,
    contrast: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blends positive values without imposing a probability upper bound."""

    midpoint = 0.5 * (a + b)
    return (
        np.maximum(midpoint + contrast * (a - midpoint), 1e-8),
        np.maximum(midpoint + contrast * (b - midpoint), 1e-8),
    )


def build_population(config: IntuitiveConfig = IntuitiveConfig()) -> SyntheticPopulation:
    """Builds one stable eligible population used by every campaign."""

    if not 0.0 <= config.activity_contrast <= 1.5:
        raise ValueError("activity_contrast must be between 0 and 1.5")
    if not 0.0 <= config.response_contrast <= 1.5:
        raise ValueError("response_contrast must be between 0 and 1.5")
    if not 0.0 <= config.profile_strength <= 1.25:
        raise ValueError("profile_strength must be between 0 and 1.25")
    for name, value in (
        ("reference_coverage_a", config.reference_coverage_a),
        ("reference_coverage_b", config.reference_coverage_b),
        ("reference_agreement", config.reference_agreement),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be greater than zero and at most one")
    shares = np.asarray([row.share for row in SEGMENTS])
    if not np.isclose(shares.sum(), 1.0):
        raise ValueError("Segment shares must sum to one")

    rng = np.random.default_rng(config.random_seed)
    segment = rng.choice(len(SEGMENTS), config.population_size, p=shares)
    common = rng.lognormal(0.0, 0.16, config.population_size)
    publisher_a = rng.lognormal(0.0, 0.12, config.population_size)
    publisher_b = rng.lognormal(0.0, 0.12, config.population_size)
    response_a = rng.lognormal(0.0, 0.22, config.population_size)
    response_b = rng.lognormal(0.0, 0.22, config.population_size)
    cost_noise = rng.lognormal(0.0, 0.10, config.population_size)

    raw_feed_a = np.clip(_values("feed_a", segment) * common * publisher_a, 0.005, 1.0)
    raw_feed_b = np.clip(_values("feed_b", segment) * common * publisher_b, 0.005, 1.0)
    raw_video_a = np.clip(_values("video_a", segment) * common * publisher_a, 0.005, 1.0)
    raw_video_b = np.clip(_values("video_b", segment) * common * publisher_b, 0.005, 1.0)
    feed_a, feed_b = _blend_pair(raw_feed_a, raw_feed_b, config.activity_contrast)
    video_a, video_b = _blend_pair(raw_video_a, raw_video_b, config.activity_contrast)

    raw_click_a = np.clip(_values("click_a", segment) * response_a, 1e-4, 1.0)
    raw_click_b = np.clip(_values("click_b", segment) * response_b, 1e-4, 1.0)
    click_a, click_b = _blend_pair(raw_click_a, raw_click_b, config.response_contrast)

    raw_cost_a = _values("cost_a", segment) * cost_noise
    raw_cost_b = _values("cost_b", segment) * cost_noise
    cost_a, cost_b = _blend_positive_pair(
        raw_cost_a,
        raw_cost_b,
        config.activity_contrast,
    )
    daytime_share = np.clip(
        _values("daytime_share", segment)
        * rng.lognormal(0.0, 0.08, config.population_size),
        0.03,
        0.97,
    )

    # Reference availability is sampled independently of campaign delivery.  The
    # final agreement draw represents the chance that two available identifiers
    # can be recognized as the same person across publishers.
    reference_a = rng.random(config.population_size) < config.reference_coverage_a
    reference_b = rng.random(config.population_size) < config.reference_coverage_b
    reference_agrees = (
        reference_a
        & reference_b
        & (rng.random(config.population_size) < config.reference_agreement)
    )

    return SyntheticPopulation(
        segment=segment,
        feed_a=feed_a,
        feed_b=feed_b,
        video_a=video_a,
        video_b=video_b,
        daytime_share=daytime_share,
        click_a=click_a,
        click_b=click_b,
        cost_a=cost_a,
        cost_b=cost_b,
        reference_a=reference_a,
        reference_b=reference_b,
        reference_agrees=reference_agrees,
    )


def make_campaign_plans(config: IntuitiveConfig = IntuitiveConfig()) -> list[CampaignPlan]:
    """Creates matched tests without campaign-specific latent-population shocks."""

    plans: list[CampaignPlan] = []
    for ordinal in range(config.training_replicates):
        plans.append(
            CampaignPlan(
                f"train-reach-{ordinal + 1:02d}",
                f"train-{ordinal + 1:02d}",
                "canonical_large_reach",
                "train",
                "reach",
                "broad_long",
                config.large_reach,
                config.large_reach,
            )
        )

    objective_profiles = ("broad_long", "feed_short", "video_short", "a_feed_b_video")
    for profile in objective_profiles:
        for ordinal in range(config.objective_replicates_per_profile):
            comparison = f"objective-{profile}-{ordinal + 1:02d}"
            for objective in ("reach", "traffic"):
                plans.append(
                    CampaignPlan(
                        f"objective-{objective}-{profile}-{ordinal + 1:02d}",
                        comparison,
                        f"objective_{objective}",
                        "evaluation",
                        objective,
                        profile,
                        config.large_reach,
                        config.large_reach,
                    )
                )

    for ordinal in range(config.direction_replicates):
        comparison = f"direction-{ordinal + 1:02d}"
        for scenario, reach_a, reach_b in (
            ("direction_medium_large", config.medium_reach, config.large_reach),
            ("direction_large_medium", config.large_reach, config.medium_reach),
        ):
            plans.append(
                CampaignPlan(
                    f"{scenario}-{ordinal + 1:02d}",
                    comparison,
                    scenario,
                    "evaluation",
                    "reach",
                    "broad_long",
                    reach_a,
                    reach_b,
                )
            )

    for profile in PROFILE_BY_NAME:
        for ordinal in range(config.large_profile_replicates):
            plans.append(
                CampaignPlan(
                    f"large-profile-{profile}-{ordinal + 1:02d}",
                    f"large-profile-{profile}-{ordinal + 1:02d}",
                    "large_reach_profiles",
                    "evaluation",
                    "reach",
                    profile,
                    config.large_reach,
                    config.large_reach,
                )
            )
    return plans


def _seed(random_seed: int, comparison_id: str) -> np.random.SeedSequence:
    return np.random.SeedSequence([random_seed, *comparison_id.encode("utf-8")])


def _profile_activity(
    population: SyntheticPopulation,
    profile: CampaignProfile,
    publisher: str,
    config: IntuitiveConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if publisher == "a":
        feed, video = population.feed_a, population.video_a
        video_share = profile.video_share_a
        click, cost = population.click_a, population.cost_a
    elif publisher == "b":
        feed, video = population.feed_b, population.video_b
        video_share = profile.video_share_b
        click, cost = population.click_b, population.cost_b
    else:
        raise ValueError(f"Unknown publisher {publisher!r}")

    placement_activity = (1.0 - video_share) * feed + video_share * video
    generic_activity = 0.5 * (feed + video)
    activity = generic_activity + config.profile_strength * (placement_activity - generic_activity)
    if profile.schedule == "daytime":
        schedule = 0.35 + 1.30 * population.daytime_share
    elif profile.schedule == "evening":
        schedule = 0.35 + 1.30 * (1.0 - population.daytime_share)
    elif profile.schedule == "broad":
        schedule = np.ones_like(activity)
    else:
        raise ValueError(f"Unknown schedule {profile.schedule!r}")
    schedule = 1.0 + config.profile_strength * (schedule - 1.0)
    effective_days = _effective_active_days(profile, config.profile_strength)
    opportunity = 1.0 - np.exp(-0.075 * effective_days * activity * schedule)
    return np.clip(opportunity, 1e-8, 1.0), click, cost


def _effective_active_days(
    profile: CampaignProfile,
    profile_strength: float,
) -> float:
    """Interpolates flight length while requiring a meaningful positive duration."""

    canonical_days = PROFILE_BY_NAME["broad_long"].active_days
    effective_days = canonical_days + profile_strength * (
        profile.active_days - canonical_days
    )
    if effective_days <= 0.0:
        raise ValueError(
            f"Profile {profile.name!r} has nonpositive effective active days"
        )
    return float(effective_days)


def _delivery_weight(
    population: SyntheticPopulation,
    plan: CampaignPlan,
    publisher: str,
    config: IntuitiveConfig,
) -> np.ndarray:
    opportunity, click, cost = _profile_activity(
        population,
        PROFILE_BY_NAME[plan.profile],
        publisher,
        config,
    )
    if plan.objective == "reach":
        objective_value = np.ones_like(opportunity)
    elif plan.objective == "traffic":
        objective_value = np.power(click, config.traffic_response_exponent)
    else:
        raise ValueError(f"Unknown objective {plan.objective!r}")
    return np.maximum(opportunity * objective_value / cost, 1e-10)


def _select(
    uniform_gumbel: np.ndarray,
    weights: np.ndarray,
    count: int,
    concentration: float,
) -> np.ndarray:
    if not 0 < count < len(weights):
        raise ValueError("Reach must be positive and smaller than population")
    keys = concentration * np.log(weights) + uniform_gumbel
    return np.argpartition(keys, -count)[-count:]


def simulate_campaign(
    population: SyntheticPopulation,
    plan: CampaignPlan,
    config: IntuitiveConfig = IntuitiveConfig(),
) -> CampaignResult:
    """Selects publisher audiences independently and measures their person overlap."""

    rng = np.random.default_rng(_seed(config.random_seed, plan.comparison_id))
    gumbel_a = rng.gumbel(size=config.population_size)
    gumbel_b = rng.gumbel(size=config.population_size)
    selected_a = _select(
        gumbel_a,
        _delivery_weight(population, plan, "a", config),
        plan.reach_a,
        config.selection_concentration,
    )
    selected_b = _select(
        gumbel_b,
        _delivery_weight(population, plan, "b", config),
        plan.reach_b,
        config.selection_concentration,
    )
    reached_a = np.zeros(config.population_size, dtype=bool)
    reached_b = np.zeros(config.population_size, dtype=bool)
    reached_a[selected_a] = True
    reached_b[selected_b] = True
    overlap = int(np.count_nonzero(reached_a & reached_b))
    smaller = min(plan.reach_a, plan.reach_b)
    reference_matches = int(
        np.count_nonzero(reached_a & reached_b & population.reference_agrees)
    )
    reference_observation_probability = (
        config.reference_coverage_a
        * config.reference_coverage_b
        * config.reference_agreement
    )
    reference_overlap_estimate = float(
        np.clip(
            reference_matches / (reference_observation_probability * smaller),
            0.0,
            1.0,
        )
    )
    direction = (
        "large_large"
        if plan.reach_a == plan.reach_b
        else ("medium_large" if plan.reach_a < plan.reach_b else "large_medium")
    )
    return CampaignResult(
        campaign_id=plan.campaign_id,
        comparison_id=plan.comparison_id,
        scenario=plan.scenario,
        split=plan.split,
        objective=plan.objective,
        profile=plan.profile,
        direction=direction,
        reach_a=plan.reach_a,
        reach_b=plan.reach_b,
        overlap=overlap,
        overlap_rate=overlap / smaller,
        union_reach=plan.reach_a + plan.reach_b - overlap,
        reference_matches=reference_matches,
        reference_observation_probability=reference_observation_probability,
        reference_overlap_estimate=reference_overlap_estimate,
    )


def simulate_market(config: IntuitiveConfig = IntuitiveConfig()) -> list[CampaignResult]:
    population = build_population(config)
    return [simulate_campaign(population, plan, config) for plan in make_campaign_plans(config)]


def diagnostics(campaigns: Iterable[CampaignResult]) -> dict[str, object]:
    rows = list(campaigns)
    if not rows:
        raise ValueError("At least one campaign is required")
    evaluation = [row for row in rows if row.split == "evaluation"]
    by_comparison: dict[str, dict[str, CampaignResult]] = {}
    for row in evaluation:
        by_comparison.setdefault(row.comparison_id, {})[row.scenario] = row

    objective_delta = np.asarray(
        [
            pair["objective_traffic"].overlap_rate - pair["objective_reach"].overlap_rate
            for pair in by_comparison.values()
            if {"objective_traffic", "objective_reach"}.issubset(pair)
        ]
    )
    direction_delta = np.asarray(
        [
            pair["direction_medium_large"].overlap_rate
            - pair["direction_large_medium"].overlap_rate
            for pair in by_comparison.values()
            if {"direction_medium_large", "direction_large_medium"}.issubset(pair)
        ]
    )
    large = [row for row in evaluation if row.scenario == "large_reach_profiles"]
    large_overlap = np.asarray([row.overlap_rate for row in large])
    profile_rows = []
    for profile in PROFILE_BY_NAME:
        values = np.asarray([row.overlap_rate for row in large if row.profile == profile])
        profile_rows.append(
            {
                "profile": profile,
                "label": PROFILE_BY_NAME[profile].label,
                "campaigns": len(values),
                "median_overlap_percent": float(100.0 * np.median(values)),
                "p10_overlap_percent": float(100.0 * np.quantile(values, 0.10)),
                "p90_overlap_percent": float(100.0 * np.quantile(values, 0.90)),
            }
        )

    training = [row.overlap_rate for row in rows if row.split == "train"]
    baseline = float(np.mean(training))
    evaluation_error = np.asarray([100.0 * abs(row.overlap_rate - baseline) for row in evaluation])
    reference_error = np.asarray(
        [
            100.0 * abs(row.overlap_rate - row.reference_overlap_estimate)
            for row in evaluation
        ]
    )
    calibration_groups = {
        "Matched Reach": [
            row for row in evaluation if row.scenario == "objective_reach"
        ],
        "Matched Traffic": [
            row for row in evaluation if row.scenario == "objective_traffic"
        ],
        "A medium → B large": [
            row for row in evaluation if row.scenario == "direction_medium_large"
        ],
        "A large → B medium": [
            row for row in evaluation if row.scenario == "direction_large_medium"
        ],
        "Large Reach profiles": [
            row for row in evaluation if row.scenario == "large_reach_profiles"
        ],
    }
    calibration_rows = []
    for label, group in calibration_groups.items():
        actual = np.asarray([row.overlap_rate for row in group])
        reference = np.asarray([row.reference_overlap_estimate for row in group])
        fixed_error = 100.0 * np.abs(actual - baseline)
        calibrated_error = 100.0 * np.abs(actual - reference)
        smaller = np.asarray(
            [min(row.reach_a, row.reach_b) for row in group], dtype=float
        )
        total_reach = np.asarray(
            [row.reach_a + row.reach_b for row in group], dtype=float
        )
        actual_union = total_reach - actual * smaller
        fixed_union = total_reach - baseline * smaller
        reference_union = total_reach - reference * smaller
        actual_incremental = (1.0 - actual) * smaller
        fixed_incremental = (1.0 - baseline) * smaller
        reference_incremental = (1.0 - reference) * smaller
        fixed_incremental_relative_error = (
            100.0
            * np.abs(fixed_incremental - actual_incremental)
            / actual_incremental
        )
        reference_incremental_relative_error = (
            100.0
            * np.abs(reference_incremental - actual_incremental)
            / actual_incremental
        )
        calibration_rows.append(
            {
                "group": label,
                "campaigns": len(group),
                "actual_median_overlap_percent": float(100.0 * np.median(actual)),
                "fixed_overlap_percent": float(100.0 * baseline),
                "reference_median_overlap_percent": float(
                    100.0 * np.median(reference)
                ),
                "fixed_mae_points": float(np.mean(fixed_error)),
                "reference_mae_points": float(np.mean(calibrated_error)),
                "fixed_p90_error_points": float(np.quantile(fixed_error, 0.90)),
                "reference_p90_error_points": float(
                    np.quantile(calibrated_error, 0.90)
                ),
                "fixed_union_mape_percent": float(
                    100.0 * np.mean(np.abs(fixed_union - actual_union) / actual_union)
                ),
                "reference_union_mape_percent": float(
                    100.0
                    * np.mean(np.abs(reference_union - actual_union) / actual_union)
                ),
                "fixed_incremental_unique_mape_percent": float(
                    np.mean(fixed_incremental_relative_error)
                ),
                "reference_incremental_unique_mape_percent": float(
                    np.mean(reference_incremental_relative_error)
                ),
                "fixed_incremental_unique_p90_error_percent": float(
                    np.quantile(fixed_incremental_relative_error, 0.90)
                ),
                "reference_incremental_unique_p90_error_percent": float(
                    np.quantile(reference_incremental_relative_error, 0.90)
                ),
            }
        )
    p10 = float(100.0 * np.quantile(large_overlap, 0.10))
    p90 = float(100.0 * np.quantile(large_overlap, 0.90))
    return {
        "baseline_overlap_percent": 100.0 * baseline,
        "objective": {
            "pairs": len(objective_delta),
            "traffic_lower_share": float(np.mean(objective_delta < 0.0)),
            "traffic_minus_reach_median_points": float(100.0 * np.median(objective_delta)),
            "p10_points": float(100.0 * np.quantile(objective_delta, 0.10)),
            "p90_points": float(100.0 * np.quantile(objective_delta, 0.90)),
        },
        "direction": {
            "pairs": len(direction_delta),
            "medium_large_higher_share": float(np.mean(direction_delta > 0.0)),
            "medium_large_minus_large_medium_median_points": float(100.0 * np.median(direction_delta)),
            "p10_points": float(100.0 * np.quantile(direction_delta, 0.10)),
            "p90_points": float(100.0 * np.quantile(direction_delta, 0.90)),
        },
        "large_profiles": {
            "campaigns": len(large_overlap),
            "p10_overlap_percent": p10,
            "median_overlap_percent": float(100.0 * np.median(large_overlap)),
            "p90_overlap_percent": p90,
            "p10_p90_width_points": p90 - p10,
            "profiles": profile_rows,
        },
        "global_model": {
            "evaluation_mae_points": float(np.mean(evaluation_error)),
            "evaluation_p90_error_points": float(np.quantile(evaluation_error, 0.90)),
        },
        "reference_calibration": {
            "observation_probability_percent": float(
                100.0 * rows[0].reference_observation_probability
            ),
            "evaluation_mae_points": float(np.mean(reference_error)),
            "evaluation_p90_error_points": float(
                np.quantile(reference_error, 0.90)
            ),
            "groups": calibration_rows,
        },
    }


def run_experiment(
    output_directory: str | Path,
    config: IntuitiveConfig = IntuitiveConfig(),
) -> tuple[list[CampaignResult], dict[str, object]]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    campaigns = simulate_market(config)
    report = {"config": asdict(config), "diagnostics": diagnostics(campaigns)}
    _write_csv(output / "segments.csv", [asdict(row) for row in SEGMENTS])
    _write_csv(output / "profiles.csv", [asdict(row) for row in PROFILES])
    _write_csv(output / "campaigns.csv", [asdict(row) for row in campaigns])
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return campaigns, report


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
