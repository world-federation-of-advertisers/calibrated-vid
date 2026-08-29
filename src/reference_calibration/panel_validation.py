from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .evaluation import CalibratedReport, relative_error, summarize
from .joint_decoding import calibrate_report_pairwise_maximum_entropy
from .measurement import ReportObservation, calibration_dataset, measure_report
from .models import CalibrationModel, LatentMixtureModel, PairAwareLogModel
from .population import (
    AUDIENCE_STRATEGIES,
    CAMPAIGN_OBJECTIVES,
    META_CAMPAIGN_SCENARIOS,
    META_SCENARIO_DESCRIPTIONS,
    Campaign,
    SyntheticWorld,
    generate_campaign,
    make_world,
)
from .provider_model import (
    ContextualDemographicAllocator,
    ProportionalDemographicAllocator,
    demographic_distribution_error,
    demographic_reach_error,
)
from .sets import (
    inclusive_intersections,
    members,
    project_to_bounded_sum,
    union_values,
)


PANEL_DESIGNS = {
    "representative": {
        "label": "Representative panel",
        "description": "A simple random panel. Its main limitation is sampling noise.",
    },
    "low_effective_size": {
        "label": "5,000 members, lower effective size",
        "description": "The panel has 5,000 members but unequal survey weights reduce the information content.",
    },
    "observable_bias_weighted": {
        "label": "Observable bias, weighted",
        "description": "Age and geography affect recruitment, and known selection weights correct most of the bias.",
    },
    "hidden_matchability_bias": {
        "label": "Hidden intent/matchability bias",
        "description": "Recruitment also favors high-intent, email-matchable people; ordinary demographic weights cannot remove it.",
    },
}


@dataclass(frozen=True)
class PanelSample:
    design: str
    indices: np.ndarray
    weights: np.ndarray
    effective_size: float

    @property
    def raw_size(self) -> int:
        return int(len(self.indices))


class _UnitCaptureModel(CalibrationModel):
    name = "unit_capture_placeholder"

    def predict_capture(self, subset_masks: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        del log_scale
        return np.ones(len(subset_masks), dtype=float)

    def describe(self) -> dict:
        return {"name": self.name, "family": "placeholder"}


UNIT_CAPTURE_MODEL = _UnitCaptureModel()


def report_specs(n_edps: int, n_weeks: int):
    all_edps = tuple(range(n_edps))
    five = tuple(range(min(5, n_edps)))
    return (
        ("weeks_1_3__2_edps", tuple(range(min(3, n_weeks))), (0, 1)),
        ("weeks_5_12__2_edps", tuple(range(4, min(12, n_weeks))), (0, 1)),
        ("all_weeks__2_edps", tuple(range(n_weeks)), (0, 1)),
        ("weeks_7_13__5_edps", tuple(range(6, n_weeks)), five),
        ("all_weeks__5_edps", tuple(range(n_weeks)), five),
        (
            "noncontiguous__5_edps",
            tuple(index for index in (0, 2, 4, 7, 10, 12) if index < n_weeks),
            five,
        ),
        ("weeks_1_3__10_edps", tuple(range(min(3, n_weeks))), all_edps),
        ("weeks_1_12__10_edps", tuple(range(min(12, n_weeks))), all_edps),
        ("all_weeks__10_edps", tuple(range(n_weeks)), all_edps),
    )


def training_specs(n_edps: int, n_weeks: int):
    wanted = {
        "weeks_1_3__2_edps",
        "all_weeks__2_edps",
        "weeks_7_13__5_edps",
        "all_weeks__5_edps",
        "weeks_1_12__10_edps",
        "all_weeks__10_edps",
    }
    return tuple(spec for spec in report_specs(n_edps, n_weeks) if spec[0] in wanted)


def _normalized_probabilities(score: np.ndarray) -> np.ndarray:
    shifted = np.asarray(score, dtype=float) - float(np.max(score))
    probabilities = np.exp(np.clip(shifted, -30.0, 0.0))
    return probabilities / float(probabilities.sum())


def draw_panel(
    world: SyntheticWorld,
    design: str,
    seed: int,
    panel_size: int | None = None,
) -> PanelSample:
    if design not in PANEL_DESIGNS:
        raise ValueError(f"unknown panel design: {design}")
    rng = np.random.default_rng(seed)
    size = min(panel_size or world.config.panel_size, world.config.n_users)
    n = world.config.n_users

    age = world.true_demographic // 6
    geo = world.true_demographic % 3
    observable_score = 0.45 * (age == 0) - 0.30 * (age == 2) + 0.35 * (geo == 2)

    if design in {"representative", "low_effective_size"}:
        selection_score = np.zeros(n, dtype=float)
        known_score = selection_score
    elif design == "observable_bias_weighted":
        selection_score = observable_score
        known_score = observable_score
    else:
        hidden_score = 0.95 * world.matchability + 0.70 * world.segments[0]
        selection_score = observable_score + hidden_score
        known_score = observable_score

    probabilities = _normalized_probabilities(selection_score)
    indices = rng.choice(n, size=size, replace=False, p=probabilities)
    if design == "representative":
        raw_weights = np.ones(size, dtype=float)
    elif design == "low_effective_size":
        # Independent unequal weights mimic a nominal panel whose effective
        # sample is smaller because a few members carry much more weight.
        raw_weights = np.exp(1.10 * world.segments[4, indices])
    else:
        known_probability = _normalized_probabilities(known_score)
        raw_weights = 1.0 / np.maximum(known_probability[indices], 1e-12)

    weights = raw_weights * world.config.population_size / float(raw_weights.sum())
    effective_size = float(weights.sum() ** 2 / np.sum(weights * weights))
    return PanelSample(design, indices.astype(np.int64), weights, effective_size)


def _weighted_exact_cells(membership: np.ndarray, weights: np.ndarray) -> np.ndarray:
    powers = (1 << np.arange(membership.shape[0], dtype=np.int64))[:, None]
    masks = np.sum(membership.astype(np.int64) * powers, axis=0)
    return np.bincount(masks, weights=weights, minlength=1 << membership.shape[0]).astype(float)


def _global_masks(edps: tuple[int, ...]) -> np.ndarray:
    result = np.zeros(1 << len(edps), dtype=np.int64)
    for local_mask in range(1, 1 << len(edps)):
        global_mask = 0
        for local_index, edp in enumerate(edps):
            if local_mask & (1 << local_index):
                global_mask |= 1 << edp
        result[local_mask] = global_mask
    return result


def measure_panel_report(
    world: SyntheticWorld,
    campaign: Campaign,
    panel: PanelSample,
    weeks: tuple[int, ...],
    edps: tuple[int, ...],
) -> ReportObservation:
    selected_events = campaign.events[np.ix_(edps, weeks, panel.indices)]
    reached = np.any(selected_events, axis=1)
    exact = _weighted_exact_cells(reached, panel.weights)
    intersections = inclusive_intersections(exact)
    unions = union_values(exact)
    population = float(panel.weights.sum())
    marginals = np.asarray(
        [intersections[1 << local] for local in range(len(edps))],
        dtype=float,
    )
    reach_fractions = marginals / population
    size = 1 << len(edps)
    baseline_intersections = np.zeros(size, dtype=float)
    baseline_unions = np.zeros(size, dtype=float)
    for subset in range(1, size):
        local = members(subset, len(edps))
        fractions = reach_fractions[list(local)]
        if len(local) == 1:
            baseline_intersections[subset] = marginals[local[0]]
        else:
            baseline_intersections[subset] = population * float(np.prod(fractions))
        baseline_unions[subset] = population * (1.0 - float(np.prod(1.0 - fractions)))

    visible = reached & world.email_linkable[np.asarray(edps)][:, panel.indices]
    visible_exact = _weighted_exact_cells(visible, panel.weights)
    reference_signal = inclusive_intersections(visible_exact)
    reference = reference_signal.copy()
    collision_floor = np.zeros(size, dtype=float)

    demographic_count = len(world.demographic_labels)
    true_demo = world.true_demographic[panel.indices]
    vid_demo = world.vid_demographic[panel.indices]
    demographic_population = np.bincount(
        true_demo,
        weights=panel.weights,
        minlength=demographic_count,
    ).astype(float)
    any_reached = np.any(reached, axis=0)
    truth_demographic_union = np.bincount(
        true_demo[any_reached],
        weights=panel.weights[any_reached],
        minlength=demographic_count,
    ).astype(float)
    edp_demographic_reaches = np.zeros((len(edps), demographic_count), dtype=float)
    for local in range(len(edps)):
        edp_demographic_reaches[local] = np.bincount(
            vid_demo[reached[local]],
            weights=panel.weights[reached[local]],
            minlength=demographic_count,
        ).astype(float)
    raw_demographic_union = np.zeros(demographic_count, dtype=float)
    for demographic in range(demographic_count):
        demo_population = max(float(demographic_population[demographic]), 1.0)
        fractions = np.clip(
            edp_demographic_reaches[:, demographic] / demo_population,
            0.0,
            1.0,
        )
        raw_demographic_union[demographic] = demo_population * (
            1.0 - float(np.prod(1.0 - fractions))
        )
    baseline_total = float(baseline_unions[-1])
    if float(raw_demographic_union.sum()) <= 0:
        raw_demographic_union = demographic_population.copy()
    baseline_demographic_union = project_to_bounded_sum(
        raw_demographic_union * baseline_total / max(float(raw_demographic_union.sum()), 1.0),
        baseline_total,
        upper=demographic_population,
    )

    return ReportObservation(
        campaign_id=campaign.campaign_id,
        scenario=campaign.scenario,
        weeks=tuple(weeks),
        edps=tuple(edps),
        person_weight=population / max(panel.effective_size, 1.0),
        global_masks=_global_masks(edps),
        reach_fractions=reach_fractions,
        truth_exact_cells=exact,
        truth_intersections=intersections,
        truth_unions=unions,
        baseline_intersections=baseline_intersections,
        baseline_unions=baseline_unions,
        email_intersections=reference_signal.copy(),
        reference_intersections=reference,
        collision_floor=collision_floor,
        reference_signal=reference_signal,
        objectives=tuple(campaign.objectives[edp] for edp in edps),
        audience_strategies=tuple(campaign.audience_strategies[edp] for edp in edps),
        demographic_labels=world.demographic_labels,
        demographic_population=demographic_population,
        edp_demographic_reaches=edp_demographic_reaches,
        truth_demographic_union=truth_demographic_union,
        baseline_demographic_union=baseline_demographic_union,
    )


def _pair_feature(
    observation: ReportObservation,
    left: int,
    right: int,
    n_edps: int,
    include_context: bool,
) -> np.ndarray:
    pair_list = tuple(combinations(range(n_edps), 2))
    global_pair = tuple(sorted((observation.edps[left], observation.edps[right])))
    values = [float(global_pair == pair) for pair in pair_list]
    if include_context:
        objectives = (observation.objectives[left], observation.objectives[right])
        strategies = (
            observation.audience_strategies[left],
            observation.audience_strategies[right],
        )
        values.extend(
            0.5 * sum(item == label for item in objectives)
            for label in CAMPAIGN_OBJECTIVES[:-1]
        )
        values.extend(
            0.5 * sum(item == label for item in strategies)
            for label in AUDIENCE_STRATEGIES[:-1]
        )
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class EmailFirstPanelVidModel:
    """Aggregate surrogate for an email-first, demographic-agnostic VID labeler.

    Email and proprietary identifiers are separate labeler inputs.  Shared
    email deterministically anchors the same VID across EDPs.  The fitted
    response estimates only the remaining overlap that must be assigned from
    proprietary identifiers, co-viewing, and other non-email cases.  It may
    use EDP identity and approved impression-level campaign context, but it
    never consumes the downstream Reference-ID measurement, report scale, or
    aggregate Reference-ID intersections.
    """

    n_edps: int
    include_context: bool
    coefficients: np.ndarray
    ridge_penalty: float

    @property
    def parameter_count(self) -> int:
        return int(len(self.coefficients))

    def describe(self) -> dict:
        return {
            "name": "email_first_demographic_agnostic_vid",
            "model_type": "impression_level_vid_labeler_surrogate",
            "identity_inputs": ["optional_normalized_email", "optional_edp_proprietary_id"],
            "optional_context": (
                ["campaign_objective", "audience_strategy", "co_viewing_or_other_context"]
                if self.include_context
                else []
            ),
            "uses_reference_id_calibration_input": False,
            "n_edps": self.n_edps,
            "parameter_count": self.parameter_count,
            "ridge_penalty": self.ridge_penalty,
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def fit(
        cls,
        observations: list[ReportObservation],
        n_edps: int,
        include_context: bool = True,
        ridge_penalty: float = 2.0,
    ) -> "EmailFirstPanelVidModel":
        features: list[np.ndarray] = []
        targets: list[float] = []
        weights: list[float] = []
        campaign_ids: list[str] = []
        for observation in observations:
            for left, right in combinations(range(len(observation.edps)), 2):
                mask = (1 << left) | (1 << right)
                left_reach = float(observation.truth_intersections[1 << left])
                right_reach = float(observation.truth_intersections[1 << right])
                population = float(observation.truth_intersections[0])
                lower = max(left_reach + right_reach - population, 0.0)
                upper = min(left_reach, right_reach)
                email_anchor = float(observation.email_intersections[mask])
                anchor = float(np.clip(max(lower, email_anchor), lower, upper))
                remaining_capacity = upper - anchor
                if remaining_capacity <= 0:
                    continue
                truth = float(observation.truth_intersections[mask])
                capture = (truth - anchor) / remaining_capacity
                capture = float(np.clip(capture, 1e-4, 1.0 - 1e-4))
                features.append(_pair_feature(observation, left, right, n_edps, include_context))
                targets.append(float(np.log(capture / (1.0 - capture))))
                effective = remaining_capacity / max(observation.person_weight, 1.0)
                weights.append(float(np.sqrt(max(effective, 1.0))))
                campaign_ids.append(observation.campaign_id)
        matrix = np.vstack(features)
        response = np.asarray(targets, dtype=float)
        fit_weight = np.asarray(weights, dtype=float)
        fit_weight /= max(float(np.median(fit_weight)), 1.0)
        fit_weight = np.clip(fit_weight, 0.25, 4.0)
        for campaign_id in set(campaign_ids):
            selected = np.asarray([value == campaign_id for value in campaign_ids])
            norm = float(np.linalg.norm(fit_weight[selected]))
            if norm > 0:
                fit_weight[selected] /= norm
        penalty = np.full(matrix.shape[1], ridge_penalty, dtype=float)
        penalty[: len(tuple(combinations(range(n_edps), 2)))] *= 0.20
        augmented_matrix = np.vstack(
            [matrix * fit_weight[:, None], np.diag(np.sqrt(penalty))]
        )
        augmented_response = np.concatenate(
            [response * fit_weight, np.zeros(matrix.shape[1], dtype=float)]
        )
        coefficients, *_ = np.linalg.lstsq(
            augmented_matrix,
            augmented_response,
            rcond=None,
        )
        return cls(n_edps, include_context, coefficients, ridge_penalty)

    def predict_pair_targets(self, observation: ReportObservation) -> np.ndarray:
        target = observation.baseline_intersections.copy()
        population = float(observation.truth_intersections[0])
        for left, right in combinations(range(len(observation.edps)), 2):
            mask = (1 << left) | (1 << right)
            feature = _pair_feature(
                observation,
                left,
                right,
                self.n_edps,
                self.include_context,
            )
            logit = float(np.clip(feature @ self.coefficients, -12.0, 12.0))
            capture = 1.0 / (1.0 + np.exp(-logit))
            left_reach = float(observation.truth_intersections[1 << left])
            right_reach = float(observation.truth_intersections[1 << right])
            lower = max(left_reach + right_reach - population, 0.0)
            upper = min(left_reach, right_reach)
            email_anchor = float(observation.email_intersections[mask])
            anchor = float(np.clip(max(lower, email_anchor), lower, upper))
            target[mask] = float(
                np.clip(anchor + capture * (upper - anchor), lower, upper)
            )
        return target

    def predict_report(self, observation: ReportObservation) -> CalibratedReport:
        return calibrate_report_pairwise_maximum_entropy(
            observation,
            UNIT_CAPTURE_MODEL,
            pair_ridge=1e-6,
            evidence_half_saturation=0.0,
            name="panel_trained_email_first_demographic_agnostic_vid",
            pair_target_intersections=self.predict_pair_targets(observation),
        )


@dataclass(frozen=True)
class TwoVidAggregateCombiner:
    """Provider-trained combination of two VID models' aggregate overlaps.

    The combiner never links identifiers across the two VID spaces.  It learns
    one bounded weight from panel campaigns and applies that weight to the two
    models' pair-intersection estimates before decoding one valid audience.
    Weight zero returns the demographic VID overlap model; weight one returns
    the demographic-agnostic model; an interior value uses both.
    """

    agnostic_weight: float

    @property
    def parameter_count(self) -> int:
        return 1

    def describe(self) -> dict:
        return {
            "name": "two_vid_aggregate_combiner",
            "model_type": "aggregate_pair_intersection_blend",
            "agnostic_weight": self.agnostic_weight,
            "demographic_vid_weight": 1.0 - self.agnostic_weight,
            "uses_person_level_crosswalk": False,
            "parameter_count": self.parameter_count,
        }

    @classmethod
    def fit(
        cls,
        observations: list[ReportObservation],
        agnostic_model: EmailFirstPanelVidModel,
    ) -> "TwoVidAggregateCombiner":
        deltas: list[float] = []
        residuals: list[float] = []
        weights: list[float] = []
        campaign_ids: list[str] = []
        for observation in observations:
            agnostic = agnostic_model.predict_pair_targets(observation)
            for left, right in combinations(range(len(observation.edps)), 2):
                mask = (1 << left) | (1 << right)
                existing = float(observation.baseline_intersections[mask])
                truth = float(observation.truth_intersections[mask])
                deltas.append(float(agnostic[mask]) - existing)
                residuals.append(truth - existing)
                effective = truth / max(observation.person_weight, 1.0)
                weights.append(float(np.sqrt(max(effective, 1.0))))
                campaign_ids.append(observation.campaign_id)
        delta = np.asarray(deltas, dtype=float)
        residual = np.asarray(residuals, dtype=float)
        fit_weight = np.asarray(weights, dtype=float)
        fit_weight /= max(float(np.median(fit_weight)), 1.0)
        fit_weight = np.clip(fit_weight, 0.25, 4.0)
        for campaign_id in set(campaign_ids):
            selected = np.asarray([value == campaign_id for value in campaign_ids])
            norm = float(np.linalg.norm(fit_weight[selected]))
            if norm > 0:
                fit_weight[selected] /= norm
        weighted_delta = fit_weight * delta
        denominator = float(weighted_delta @ weighted_delta)
        if denominator <= 1e-12:
            return cls(agnostic_weight=0.0)
        numerator = float(weighted_delta @ (fit_weight * residual))
        return cls(agnostic_weight=float(np.clip(numerator / denominator, 0.0, 1.0)))

    def predict_pair_targets(
        self,
        observation: ReportObservation,
        agnostic_model: EmailFirstPanelVidModel,
    ) -> np.ndarray:
        agnostic = agnostic_model.predict_pair_targets(observation)
        target = observation.baseline_intersections.copy()
        for left, right in combinations(range(len(observation.edps)), 2):
            mask = (1 << left) | (1 << right)
            existing = float(observation.baseline_intersections[mask])
            target[mask] = existing + self.agnostic_weight * (
                float(agnostic[mask]) - existing
            )
        return target

    def predict_report(
        self,
        observation: ReportObservation,
        agnostic_model: EmailFirstPanelVidModel,
    ) -> CalibratedReport:
        return calibrate_report_pairwise_maximum_entropy(
            observation,
            UNIT_CAPTURE_MODEL,
            pair_ridge=1e-6,
            evidence_half_saturation=0.0,
            name="provider_combined_two_vid_result",
            pair_target_intersections=self.predict_pair_targets(
                observation,
                agnostic_model,
            ),
        )


def _with_baseline(
    observation: ReportObservation,
    provider_report: CalibratedReport,
) -> ReportObservation:
    return replace(
        observation,
        baseline_intersections=provider_report.raw_intersections.copy(),
        baseline_unions=provider_report.union_values.copy(),
    )


def _campaigns(
    world: SyntheticWorld,
    per_scenario: int,
    seed_offset: int,
    prefix: str,
) -> list[Campaign]:
    result = []
    for scenario_index, scenario in enumerate(META_CAMPAIGN_SCENARIOS):
        for replicate in range(per_scenario):
            result.append(
                generate_campaign(
                    world,
                    scenario,
                    world.config.seed + seed_offset + scenario_index * 10_000 + replicate,
                    f"{prefix}_{scenario}_{replicate:02d}",
                )
            )
    return result


def _observations(
    world: SyntheticWorld,
    campaigns: list[Campaign],
    specs,
) -> dict[tuple[str, str], ReportObservation]:
    return {
        (campaign.campaign_id, label): measure_report(world, campaign, weeks, edps)
        for campaign in campaigns
        for label, weeks, edps in specs
    }


def _panel_observations(
    world: SyntheticWorld,
    campaigns: list[Campaign],
    specs,
    panel: PanelSample,
) -> dict[tuple[str, str], ReportObservation]:
    return {
        (campaign.campaign_id, label): measure_panel_report(
            world,
            campaign,
            panel,
            weeks,
            edps,
        )
        for campaign in campaigns
        for label, weeks, edps in specs
    }


def _fit_reference_models(
    config: SimulationConfig,
    panel_observations: list[ReportObservation],
    seed_offset: int,
):
    # The provider knows weighted person-level truth for its panel and observes
    # aggregate Reference-ID intersections for those same panelists. Replacing
    # K0 with panel truth makes signal / K0 the panel-estimated capture rate.
    # No full-population Reference IDs or person-level crosswalk are used.
    observations = [
        replace(
            observation,
            baseline_intersections=observation.truth_intersections.copy(),
            baseline_unions=observation.truth_unions.copy(),
        )
        for observation in panel_observations
    ]
    dataset = calibration_dataset(
        observations,
        config.minimum_calibration_intersection,
    )
    return {
        "fixed": PairAwareLogModel.fit(
            dataset,
            config.n_edps,
            "none",
            config.ridge_penalty,
        ),
        "fixed_log": PairAwareLogModel.fit(
            dataset,
            config.n_edps,
            "shared",
            config.ridge_penalty,
        ),
        "mixture": LatentMixtureModel.fit(
            dataset,
            config.n_edps,
            config.seed + 50_000 + seed_offset,
        ),
    }


def _method_totals(
    observation: ReportObservation,
    provider: EmailFirstPanelVidModel,
    combiner: TwoVidAggregateCombiner,
    reference_models: dict[str, CalibrationModel],
):
    agnostic_report = provider.predict_report(observation)
    two_vid_report = combiner.predict_report(observation, provider)
    two_vid_observation = _with_baseline(observation, two_vid_report)
    existing_reports = {}
    two_vid_reports = {}
    for name, model in reference_models.items():
        existing_reports[name] = calibrate_report_pairwise_maximum_entropy(
            observation,
            model,
            pair_ridge=1e-6,
            evidence_half_saturation=20.0,
            name=f"existing_plus_{name}",
        )
        two_vid_reports[name] = calibrate_report_pairwise_maximum_entropy(
            two_vid_observation,
            model,
            pair_ridge=1e-6,
            evidence_half_saturation=20.0,
            name=f"two_vid_plus_{name}",
        )
    totals = {
        "existing_vid": float(observation.baseline_unions[-1]),
        "agnostic_vid_diagnostic": agnostic_report.full_union,
        "two_vid": two_vid_report.full_union,
        **{
            f"existing_plus_{name}": report.full_union
            for name, report in existing_reports.items()
        },
        **{
            f"two_vid_plus_{name}": report.full_union
            for name, report in two_vid_reports.items()
        },
    }
    return totals, agnostic_report, two_vid_report, {
        "existing": existing_reports,
        "two_vid": two_vid_reports,
    }


def _calibration_instruction(
    selected_method: str,
    base_method: str,
    prefix: str,
    models: dict[str, CalibrationModel],
) -> dict:
    """Build the frozen instruction the provider sends to measurement."""
    if selected_method == base_method:
        return {
            "mode": "identity",
            "description": "Return the selected VID result without Reference-ID correction.",
        }
    family = selected_method.removeprefix(prefix)
    return {
        "mode": "active_reference_id_correction",
        "model": models[family].describe(),
        "runtime_inputs": [
            "selected_base_vid_marginal_reaches",
            "selected_base_vid_subset_intersections",
            "reference_id_subset_intersections",
            "reference_id_collision_floor",
            "population_size",
        ],
        "steps": [
            "subtract the approved collision floor from each Reference-ID intersection",
            "apply the frozen capture-rate model to the matching EDP subset",
            "decode one nonnegative audience for the requested report",
            "return bounded reach plus diagnostics",
        ],
        "decoder": {
            "family": "pairwise_maximum_entropy",
            "pair_ridge": 1e-6,
            "evidence_half_saturation": 20.0,
        },
    }


def _selected_total_reach_instruction(
    recommended: str,
    selected_existing: str,
    selected_two_vid: str,
    provider: EmailFirstPanelVidModel,
    combiner: TwoVidAggregateCombiner,
    models: dict[str, CalibrationModel],
) -> dict:
    uses_two_vids = recommended in {"two_vid", "two_vid_plus_selected_rid"}
    uses_reference_id = recommended in {
        "existing_plus_selected_rid",
        "two_vid_plus_selected_rid",
    }
    instruction = {
        "name": recommended,
        "vid_inputs": (
            ["demographic_vid", "demographic_agnostic_vid"]
            if uses_two_vids
            else ["demographic_vid"]
        ),
        "uses_reference_id": uses_reference_id,
        "uses_person_level_crosswalk": False,
    }
    if uses_two_vids:
        instruction["demographic_agnostic_vid"] = provider.describe()
        instruction["vid_output_combiner"] = combiner.describe()
    if uses_reference_id:
        selected = selected_two_vid if uses_two_vids else selected_existing
        prefix = "two_vid_plus_" if uses_two_vids else "existing_plus_"
        base = "two_vid" if uses_two_vids else "existing_vid"
        instruction["reference_id_instruction"] = _calibration_instruction(
            selected,
            base,
            prefix,
            models,
        )
    else:
        instruction["reference_id_instruction"] = {
            "mode": "disabled",
            "description": "Use the provider's VID-only finalization function.",
        }
    return instruction


def _select_active_method(
    rows: list[dict],
    base_method: str,
    candidate_methods: tuple[str, ...],
    minimum_improvement: float,
) -> str:
    by_method: dict[str, list[float]] = defaultdict(list)
    by_campaign: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_method[row["method"]].append(float(row["panel_error"]))
        by_campaign[str(row["campaign"])][row["method"]].append(float(row["panel_error"]))
    base = np.asarray(by_method[base_method], dtype=float)
    eligible: list[tuple[float, str]] = []
    for method in candidate_methods:
        candidate = np.asarray(by_method[method], dtype=float)
        mean_improvement = float(base.mean() - candidate.mean())
        p90_change = float(np.quantile(candidate, 0.90) - np.quantile(base, 0.90))
        campaign_differences = np.asarray(
            [
                np.mean(values[method]) - np.mean(values[base_method])
                for values in by_campaign.values()
                if method in values and base_method in values
            ],
            dtype=float,
        )
        standard_error = (
            float(np.std(campaign_differences, ddof=1) / np.sqrt(len(campaign_differences)))
            if len(campaign_differences) > 1
            else float("inf")
        )
        upper_90 = float(np.mean(campaign_differences) + 1.645 * standard_error)
        if (
            mean_improvement >= minimum_improvement
            and p90_change <= minimum_improvement
            and upper_90 < 0.0
        ):
            eligible.append((float(candidate.mean()), method))
    return min(eligible)[1] if eligible else base_method


def _select_configuration(
    rows: list[dict],
    candidates: tuple[str, ...],
    minimum_improvement: float,
) -> str:
    by_method: dict[str, list[float]] = defaultdict(list)
    by_campaign: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_method[row["method"]].append(float(row["panel_error"]))
        by_campaign[str(row["campaign"])][row["method"]].append(float(row["panel_error"]))
    baseline = np.asarray(by_method["existing_vid"], dtype=float)
    eligible: list[tuple[float, str]] = [(float(baseline.mean()), "existing_vid")]
    for method in candidates:
        candidate = np.asarray(by_method[method], dtype=float)
        mean_improvement = float(baseline.mean() - candidate.mean())
        p90_change = float(np.quantile(candidate, 0.90) - np.quantile(baseline, 0.90))
        campaign_differences = np.asarray(
            [
                np.mean(values[method]) - np.mean(values["existing_vid"])
                for values in by_campaign.values()
                if method in values and "existing_vid" in values
            ],
            dtype=float,
        )
        standard_error = (
            float(np.std(campaign_differences, ddof=1) / np.sqrt(len(campaign_differences)))
            if len(campaign_differences) > 1
            else float("inf")
        )
        upper_90 = float(np.mean(campaign_differences) + 1.645 * standard_error)
        if (
            mean_improvement >= minimum_improvement
            and p90_change <= minimum_improvement
            and upper_90 < 0.0
        ):
            eligible.append((float(candidate.mean()), method))
    return min(eligible)[1]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_panel_designs(rows: list[dict], path: Path) -> None:
    methods = (
        "existing_vid",
        "two_vid",
        "existing_plus_selected_rid",
        "two_vid_plus_selected_rid",
    )
    labels = {
        "existing_vid": "Existing VID",
        "two_vid": "Both VID models",
        "existing_plus_selected_rid": "Existing VID + RID",
        "two_vid_plus_selected_rid": "Both VID models + RID",
    }
    designs = tuple(PANEL_DESIGNS)
    matrix = np.zeros((len(methods), len(designs)), dtype=float)
    for method_index, method in enumerate(methods):
        for design_index, design in enumerate(designs):
            values = [
                float(row["value"])
                for row in rows
                if row["category"] == "total_error"
                and row["method"] == method
                and row["panel_design"] == design
            ]
            matrix[method_index, design_index] = float(np.mean(values))
    x = np.arange(len(designs))
    width = 0.19
    figure, axis = plt.subplots(figsize=(13, 6.5))
    for method_index, method in enumerate(methods):
        offset = (method_index - (len(methods) - 1) / 2.0) * width
        axis.bar(x + offset, 100.0 * matrix[method_index], width, label=labels[method])
    axis.set_xticks(x, [PANEL_DESIGNS[item]["label"] for item in designs], rotation=18, ha="right")
    axis.set_ylabel("Mean absolute relative error (%)")
    axis.set_title("Union-reach accuracy under four 5,000-person panel designs")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_scenarios(rows: list[dict], path: Path) -> None:
    methods = (
        "existing_vid",
        "two_vid",
        "existing_plus_selected_rid",
        "two_vid_plus_selected_rid",
    )
    labels = {
        "existing_vid": "Existing VID",
        "two_vid": "Both VID models",
        "existing_plus_selected_rid": "Existing VID + RID",
        "two_vid_plus_selected_rid": "Both VID models + RID",
    }
    scenarios = tuple(META_CAMPAIGN_SCENARIOS)
    matrix = np.zeros((len(methods), len(scenarios)), dtype=float)
    selected_rows = [row for row in rows if row["panel_design"] == "representative"]
    for method_index, method in enumerate(methods):
        for scenario_index, scenario in enumerate(scenarios):
            values = [
                float(row["value"])
                for row in selected_rows
                if row["category"] == "total_error"
                and row["method"] == method
                and row["scenario"] == scenario
            ]
            matrix[method_index, scenario_index] = float(np.mean(values))
    x = np.arange(len(scenarios))
    width = 0.19
    figure, axis = plt.subplots(figsize=(17, 7))
    for method_index, method in enumerate(methods):
        axis.bar(
            x + (method_index - (len(methods) - 1) / 2.0) * width,
            100.0 * matrix[method_index],
            width,
            label=labels[method],
        )
    axis.set_xticks(
        x,
        [
            META_SCENARIO_DESCRIPTIONS[item].get(
                "chart_label",
                META_SCENARIO_DESCRIPTIONS[item]["audience"],
            )
            for item in scenarios
        ],
        rotation=38,
        ha="right",
    )
    axis.set_ylabel("Mean absolute relative error (%)")
    axis.set_title("Representative-panel accuracy by campaign mechanism")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def run_panel_validation(
    output_dir: Path,
    profile: str = "quick",
    panel_draws: int | None = None,
):
    config = SimulationConfig.for_profile(profile)
    if panel_draws is None:
        panel_draws = config.panel_draws
    world = make_world(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_specs = training_specs(config.n_edps, config.n_weeks)
    eval_specs = report_specs(config.n_edps, config.n_weeks)

    train_per_scenario = 1 if profile == "quick" else 2
    holdout_per_scenario = 1
    evaluation_per_scenario = 1 if profile == "quick" else 2
    training_campaigns = _campaigns(world, train_per_scenario, 8_000_000, "provider_train")
    holdout_campaigns = _campaigns(world, holdout_per_scenario, 9_000_000, "provider_holdout")
    evaluation_campaigns = _campaigns(world, evaluation_per_scenario, 10_000_000, "evaluation")
    calibration_per_scenario = 1 if profile == "quick" else 2
    calibration_campaigns = _campaigns(
        world,
        calibration_per_scenario,
        11_000_000,
        "reference_calibration",
    )
    full_holdout = _observations(world, holdout_campaigns, eval_specs)
    full_evaluation = _observations(world, evaluation_campaigns, eval_specs)

    rows: list[dict] = []
    panel_rows: list[dict] = []
    activation_rows: list[dict] = []
    provider_packages: list[dict] = []
    raw_method_names = (
        "existing_vid",
        "agnostic_vid_diagnostic",
        "two_vid",
        "existing_plus_fixed",
        "existing_plus_fixed_log",
        "existing_plus_mixture",
        "two_vid_plus_fixed",
        "two_vid_plus_fixed_log",
        "two_vid_plus_mixture",
    )
    method_names = raw_method_names + (
        "existing_plus_selected_rid",
        "two_vid_plus_selected_rid",
        "provider_recommended",
    )

    for design_index, design in enumerate(PANEL_DESIGNS):
        for draw in range(panel_draws):
            panel = draw_panel(
                world,
                design,
                config.seed + 12_000_000 + design_index * 100_000 + draw,
            )
            panel_rows.append(
                {
                    "panel_design": design,
                    "draw": draw,
                    "raw_size": panel.raw_size,
                    "effective_size": panel.effective_size,
                }
            )
            panel_training = _panel_observations(
                world,
                training_campaigns,
                train_specs,
                panel,
            )
            provider = EmailFirstPanelVidModel.fit(
                list(panel_training.values()),
                config.n_edps,
                include_context=True,
            )
            proportional = ProportionalDemographicAllocator()
            contextual_demo = ContextualDemographicAllocator.fit(
                list(panel_training.values()),
                config.n_edps,
            )
            panel_calibration = _panel_observations(
                world,
                calibration_campaigns,
                train_specs,
                panel,
            )
            combiner = TwoVidAggregateCombiner.fit(
                list(panel_calibration.values()),
                provider,
            )
            reference_models = _fit_reference_models(
                config,
                list(panel_calibration.values()),
                design_index * 1_000 + draw,
            )

            panel_holdout = _panel_observations(
                world,
                holdout_campaigns,
                eval_specs,
                panel,
            )
            panel_evaluation = _panel_observations(
                world,
                evaluation_campaigns,
                eval_specs,
                panel,
            )
            holdout_decisions = []
            holdout_totals = []
            for key, full_observation in full_holdout.items():
                panel_truth = float(panel_holdout[key].truth_unions[-1])
                totals, _, _, _ = _method_totals(
                    full_observation,
                    provider,
                    combiner,
                    reference_models,
                )
                holdout_totals.append((key[0], panel_truth, totals))
                for method, estimate in totals.items():
                    holdout_decisions.append(
                        {
                            "method": method,
                            "campaign": key[0],
                            "panel_error": relative_error(estimate, panel_truth),
                        }
                    )
            selected_existing = _select_active_method(
                holdout_decisions,
                "existing_vid",
                (
                    "existing_plus_fixed",
                    "existing_plus_fixed_log",
                    "existing_plus_mixture",
                ),
                config.panel_activation_improvement,
            )
            selected_two_vid = _select_active_method(
                holdout_decisions,
                "two_vid",
                (
                    "two_vid_plus_fixed",
                    "two_vid_plus_fixed_log",
                    "two_vid_plus_mixture",
                ),
                config.panel_activation_improvement,
            )
            configuration_decisions = []
            for campaign_id, panel_truth, totals in holdout_totals:
                totals["existing_plus_selected_rid"] = totals[selected_existing]
                totals["two_vid_plus_selected_rid"] = totals[selected_two_vid]
                for method in (
                    "existing_vid",
                    "two_vid",
                    "existing_plus_selected_rid",
                    "two_vid_plus_selected_rid",
                ):
                    configuration_decisions.append(
                        {
                            "method": method,
                            "campaign": campaign_id,
                            "panel_error": relative_error(totals[method], panel_truth),
                        }
                    )
            recommended = _select_configuration(
                configuration_decisions,
                (
                    "two_vid",
                    "existing_plus_selected_rid",
                    "two_vid_plus_selected_rid",
                ),
                config.panel_activation_improvement,
            )
            provider_packages.append(
                {
                    "panel_design": design,
                    "draw": draw,
                    "selected_configuration": recommended,
                    "vid_models": {
                        "existing": {
                            "name": "existing_demographic_ready_vid",
                            "role": "demographic output plus candidate total and overlap inputs",
                        },
                        "demographic_agnostic": provider.describe(),
                    },
                    "selected_total_reach_function": _selected_total_reach_instruction(
                        recommended,
                        selected_existing,
                        selected_two_vid,
                        provider,
                        combiner,
                        reference_models,
                    ),
                    "validated_candidate_choices": {
                        "demographic_vid_plus_reference_id": selected_existing,
                        "two_vid_plus_reference_id": selected_two_vid,
                    },
                    "demographic_adjustment": {
                        "name": contextual_demo.name,
                        "instruction": (
                            "Adjust the demographic VID distribution to the selected final total; "
                            "use proportional scaling if the contextual adjustment is not validated."
                        ),
                    },
                }
            )

            draw_truth_errors: dict[str, list[float]] = defaultdict(list)
            consistency: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
            for campaign in evaluation_campaigns:
                for report_label, _, _ in eval_specs:
                    observation = full_evaluation[(campaign.campaign_id, report_label)]
                    panel_observation = panel_evaluation[(campaign.campaign_id, report_label)]
                    totals, _, two_vid_report, _ = _method_totals(
                        observation,
                        provider,
                        combiner,
                        reference_models,
                    )
                    totals["existing_plus_selected_rid"] = totals[selected_existing]
                    totals["two_vid_plus_selected_rid"] = totals[selected_two_vid]
                    totals["provider_recommended"] = totals[recommended]
                    truth_total = float(observation.truth_unions[-1])
                    truth_fraction = truth_total / float(config.population_size)
                    volume_band = (
                        "small_under_10_percent"
                        if truth_fraction < 0.10
                        else "medium_10_to_30_percent"
                        if truth_fraction < 0.30
                        else "large_over_30_percent"
                    )
                    rows.append(
                        {
                            "panel_design": design,
                            "draw": draw,
                            "scenario": campaign.scenario,
                            "campaign": campaign.campaign_id,
                            "report": report_label,
                            "edp_count": len(observation.edps),
                            "week_count": len(observation.weeks),
                            "method": "panel_estimate",
                            "selected_method": recommended,
                            "category": "panel_truth_error",
                            "value": relative_error(
                                float(panel_observation.truth_unions[-1]),
                                truth_total,
                            ),
                            "volume_band": volume_band,
                        }
                    )
                    two_vid_proportional = proportional.allocate(
                        two_vid_report.full_union,
                        observation,
                    )
                    two_vid_contextual = contextual_demo.allocate(
                        two_vid_report.full_union,
                        observation,
                    )
                    existing_rid_contextual = contextual_demo.allocate(
                        totals["existing_plus_selected_rid"],
                        observation,
                    )
                    two_vid_rid_contextual = contextual_demo.allocate(
                        totals["two_vid_plus_selected_rid"],
                        observation,
                    )
                    recommended_contextual = (
                        observation.baseline_demographic_union
                        if recommended == "existing_vid"
                        else contextual_demo.allocate(totals["provider_recommended"], observation)
                    )
                    for method in method_names:
                        value = relative_error(totals[method], truth_total)
                        draw_truth_errors[method].append(value)
                        rows.append(
                            {
                                "panel_design": design,
                                "draw": draw,
                                "scenario": campaign.scenario,
                                "campaign": campaign.campaign_id,
                                "report": report_label,
                                "edp_count": len(observation.edps),
                                "week_count": len(observation.weeks),
                                "method": method,
                                "selected_method": recommended,
                                "category": "total_error",
                                "value": value,
                                "volume_band": volume_band,
                            }
                        )
                        consistency[(campaign.campaign_id, method)][report_label] = totals[method]
                    for method, demographic in (
                        ("existing_vid", observation.baseline_demographic_union),
                        ("two_vid_proportional_demo", two_vid_proportional),
                        ("two_vid_panel_demo", two_vid_contextual),
                        ("existing_rid_panel_demo", existing_rid_contextual),
                        ("two_vid_rid_panel_demo", two_vid_rid_contextual),
                        ("recommended_panel_demo", recommended_contextual),
                    ):
                        rows.append(
                            {
                                "panel_design": design,
                                "draw": draw,
                                "scenario": campaign.scenario,
                                "campaign": campaign.campaign_id,
                                "report": report_label,
                                "edp_count": len(observation.edps),
                                "week_count": len(observation.weeks),
                                "method": method,
                                "selected_method": recommended,
                                "category": "demographic_distribution_error",
                                "value": demographic_distribution_error(
                                    demographic,
                                    observation.truth_demographic_union,
                                ),
                                "volume_band": volume_band,
                            }
                        )
                        rows.append(
                            {
                                "panel_design": design,
                                "draw": draw,
                                "scenario": campaign.scenario,
                                "campaign": campaign.campaign_id,
                                "report": report_label,
                                "edp_count": len(observation.edps),
                                "week_count": len(observation.weeks),
                                "method": method,
                                "selected_method": recommended,
                                "category": "demographic_reach_error",
                                "value": demographic_reach_error(
                                    demographic,
                                    observation.truth_demographic_union,
                                ),
                                "volume_band": volume_band,
                            }
                        )

            nested = (
                ("weeks_1_3__10_edps", "weeks_1_12__10_edps"),
                ("weeks_1_12__10_edps", "all_weeks__10_edps"),
                ("all_weeks__2_edps", "all_weeks__5_edps"),
                ("all_weeks__5_edps", "all_weeks__10_edps"),
            )
            consistency_counts = defaultdict(lambda: [0, 0])
            for (_, method), values in consistency.items():
                for smaller, larger in nested:
                    if smaller in values and larger in values:
                        consistency_counts[method][0] += 1
                        consistency_counts[method][1] += int(values[smaller] > values[larger] + 1e-6)
            existing_rid_change = float(
                np.mean(draw_truth_errors["existing_plus_selected_rid"])
                - np.mean(draw_truth_errors["existing_vid"])
            )
            two_vid_rid_change = float(
                np.mean(draw_truth_errors["two_vid_plus_selected_rid"])
                - np.mean(draw_truth_errors["two_vid"])
            )
            recommendation_change = float(
                np.mean(draw_truth_errors["provider_recommended"])
                - np.mean(draw_truth_errors["existing_vid"])
            )
            activation_rows.append(
                {
                    "panel_design": design,
                    "draw": draw,
                    "selected_existing_correction": selected_existing,
                    "selected_two_vid_correction": selected_two_vid,
                    "recommended_configuration": recommended,
                    "existing_correction_active": selected_existing != "existing_vid",
                    "two_vid_correction_active": selected_two_vid != "two_vid",
                    "existing_correction_harmed_truth": existing_rid_change > 1e-9,
                    "two_vid_correction_harmed_truth": two_vid_rid_change > 1e-9,
                    "existing_correction_error_change": existing_rid_change,
                    "two_vid_correction_error_change": two_vid_rid_change,
                    "recommended_error_change_vs_existing": recommendation_change,
                    "agnostic_model_parameter_count": provider.parameter_count,
                    "two_vid_combiner_parameter_count": combiner.parameter_count,
                    "two_vid_agnostic_weight": combiner.agnostic_weight,
                    "consistency_checks": consistency_counts["provider_recommended"][0],
                    "consistency_violations": consistency_counts["provider_recommended"][1],
                }
            )

    method_summary = {}
    for design in PANEL_DESIGNS:
        method_summary[design] = {}
        for method in method_names:
            values = [
                float(row["value"])
                for row in rows
                if row["panel_design"] == design
                and row["category"] == "total_error"
                and row["method"] == method
            ]
            method_summary[design][method] = summarize(values)

    activation_summary = {}
    for design in PANEL_DESIGNS:
        selected_rows = [row for row in activation_rows if row["panel_design"] == design]
        activation_summary[design] = {
            "draws": len(selected_rows),
            "existing_correction_active_rate": float(
                np.mean([row["existing_correction_active"] for row in selected_rows])
            ),
            "two_vid_correction_active_rate": float(
                np.mean([row["two_vid_correction_active"] for row in selected_rows])
            ),
            "existing_correction_harm_rate": float(
                np.mean([row["existing_correction_harmed_truth"] for row in selected_rows])
            ),
            "two_vid_correction_harm_rate": float(
                np.mean([row["two_vid_correction_harmed_truth"] for row in selected_rows])
            ),
            "mean_existing_correction_error_change": float(
                np.mean([row["existing_correction_error_change"] for row in selected_rows])
            ),
            "mean_two_vid_correction_error_change": float(
                np.mean([row["two_vid_correction_error_change"] for row in selected_rows])
            ),
            "mean_recommended_error_change_vs_existing": float(
                np.mean([row["recommended_error_change_vs_existing"] for row in selected_rows])
            ),
            "two_vid_agnostic_weight": summarize(
                [float(row["two_vid_agnostic_weight"]) for row in selected_rows]
            ),
            "existing_correction_selection_counts": {
                method: sum(row["selected_existing_correction"] == method for row in selected_rows)
                for method in (
                    "existing_vid",
                    "existing_plus_fixed",
                    "existing_plus_fixed_log",
                    "existing_plus_mixture",
                )
            },
            "two_vid_correction_selection_counts": {
                method: sum(row["selected_two_vid_correction"] == method for row in selected_rows)
                for method in (
                    "two_vid",
                    "two_vid_plus_fixed",
                    "two_vid_plus_fixed_log",
                    "two_vid_plus_mixture",
                )
            },
            "recommended_configuration_counts": {
                method: sum(row["recommended_configuration"] == method for row in selected_rows)
                for method in (
                    "existing_vid",
                    "two_vid",
                    "existing_plus_selected_rid",
                    "two_vid_plus_selected_rid",
                )
            },
            "raw_consistency_violation_rate": float(
                sum(row["consistency_violations"] for row in selected_rows)
                / max(sum(row["consistency_checks"] for row in selected_rows), 1)
            ),
        }

    panel_truth_summary = {}
    for design in PANEL_DESIGNS:
        panel_truth_summary[design] = {
            band: summarize(
                [
                    float(row["value"])
                    for row in rows
                    if row["panel_design"] == design
                    and row["category"] == "panel_truth_error"
                    and row["volume_band"] == band
                ]
            )
            for band in (
                "small_under_10_percent",
                "medium_10_to_30_percent",
                "large_over_30_percent",
            )
        }

    scenario_summary = {}
    for scenario in META_CAMPAIGN_SCENARIOS:
        scenario_summary[scenario] = {
            method: summarize(
                [
                    float(row["value"])
                    for row in rows
                    if row["panel_design"] == "representative"
                    and row["scenario"] == scenario
                    and row["category"] == "total_error"
                    and row["method"] == method
                ]
            )
            for method in (
                "existing_vid",
                "agnostic_vid_diagnostic",
                "two_vid",
                "existing_plus_selected_rid",
                "two_vid_plus_selected_rid",
            )
        }

    panel_summary = {
        design: {
            "raw_size": int(np.median([row["raw_size"] for row in panel_rows if row["panel_design"] == design])),
            "effective_size": summarize(
                [float(row["effective_size"]) for row in panel_rows if row["panel_design"] == design]
            ),
            **PANEL_DESIGNS[design],
        }
        for design in PANEL_DESIGNS
    }
    summary = {
        "profile": profile,
        "config": config.__dict__,
        "panel_draws_per_design": panel_draws,
        "panel_designs": panel_summary,
        "scenario_descriptions": META_SCENARIO_DESCRIPTIONS,
        "method_summary": method_summary,
        "activation_summary": activation_summary,
        "panel_truth_summary": panel_truth_summary,
        "scenario_summary": scenario_summary,
        "methods": {
            "existing_vid": "Existing demographic-ready VID architecture without Reference-ID correction.",
            "agnostic_vid_diagnostic": "Diagnostic total from the email-first demographic-agnostic VID model by itself.",
            "two_vid": "Provider-trained aggregate combination of the demographic and demographic-agnostic VID outputs, without Reference-ID input.",
            "existing_plus_fixed": "Existing VID plus the provider-fitted constant Reference-ID capture model.",
            "existing_plus_fixed_log": "Existing VID plus the provider-fitted fixed-plus-log Reference-ID model.",
            "existing_plus_mixture": "Existing VID plus the provider-fitted two-group matchability model.",
            "two_vid_plus_fixed": "Both VID outputs plus the provider-fitted constant Reference-ID model.",
            "two_vid_plus_fixed_log": "Both VID outputs plus the provider-fitted fixed-plus-log Reference-ID model.",
            "two_vid_plus_mixture": "Both VID outputs plus the provider-fitted two-group matchability model.",
            "existing_plus_selected_rid": "Existing VID plus the Reference-ID family selected by provider panel holdouts, or identity if none passes.",
            "two_vid_plus_selected_rid": "Both VID outputs plus the Reference-ID family selected by provider panel holdouts, or the VID-only function if none passes.",
            "provider_recommended": "The complete input combination and finalization function selected by provider panel holdouts.",
        },
        "provider_package_contract": {
            "provider_role": (
                "Train the optional VID model, choose how available VID outputs are combined, "
                "validate any Reference-ID input, and publish one frozen finalization function."
            ),
            "measurement_service_role": (
                "Run the supplied VID model or models, calculate aggregate Reference-ID overlaps "
                "inside the approved workload, and apply the provider's frozen instructions."
            ),
            "reference_id_is_labeler_input": False,
        },
        "report_specs": [
            {"label": label, "weeks": [week + 1 for week in weeks], "edps": [edp + 1 for edp in edps]}
            for label, weeks, edps in eval_specs
        ],
    }
    _write_csv(output_dir / "panel_validation_metrics.csv", rows)
    _write_csv(output_dir / "panel_draws.csv", panel_rows)
    _write_csv(output_dir / "activation_decisions.csv", activation_rows)
    (output_dir / "provider_packages.json").write_text(
        json.dumps(provider_packages, indent=2),
        encoding="utf-8",
    )
    (output_dir / "panel_validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _plot_panel_designs(rows, output_dir / "error_by_panel_design.png")
    _plot_scenarios(rows, output_dir / "error_by_campaign_scenario.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate provider-finalized demographic VID, two-VID, and optional "
            "Reference-ID configurations with a 5,000-person panel"
        )
    )
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--panel-draws", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/panel_5000_validation"),
    )
    arguments = parser.parse_args()
    summary = run_panel_validation(arguments.output_dir, arguments.profile, arguments.panel_draws)
    print(
        json.dumps(
            {
                "output_dir": str(arguments.output_dir.resolve()),
                "panel_draws_per_design": summary["panel_draws_per_design"],
                "panel_designs": list(summary["panel_designs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
