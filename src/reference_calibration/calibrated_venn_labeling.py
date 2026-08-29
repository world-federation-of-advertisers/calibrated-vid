from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .config import SimulationConfig
from .daily_labeling import (
    LabelingResult,
    _mix64,
    _report_union,
    _transport_cells,
    _truth_union,
    generate_temporal_stress_campaign,
)
from .evaluation import calibrate_report
from .joint_decoding import (
    JointDecoderConfig,
    calibrate_report_joint,
    calibrate_report_pairwise_maximum_entropy,
)
from .measurement import calibration_dataset, measure_report
from .models import CalibrationModel, LatentMixtureModel, PairAwareLogModel
from .panel_validation import _fit_reference_models, _panel_observations, draw_panel
from .population import Campaign, SyntheticWorld, generate_campaign, make_world
from .research_models import DirectPairLogModel
from .venn_information_proof import _proof_report_specs


TRAINING_SCENARIOS = (
    "broad_awareness_control",
    "traffic_optimization",
    "video_engagement_retargeting",
    "lead_generation",
    "sales_prospecting",
    "website_retargeting",
    "crm_customer_list",
    "catalog_retargeting",
    "lookalike_prospecting",
    "advantage_audience_expansion",
    "app_activity_retargeting",
    "unrelated_niche_control",
    "mixed_funnel_portfolio",
)

EVALUATION_SCENARIOS = TRAINING_SCENARIOS + (
    "linkage_shift_-1.0",
    "linkage_shift_0.0",
    "linkage_shift_1.0",
)


@dataclass(frozen=True)
class DecoderCandidate:
    name: str
    description: str
    decode: Callable


@dataclass(frozen=True)
class TemporalReconciliation:
    target_cells: np.ndarray
    raw_union: float
    reconciled_union: int
    union_adjustment: float
    cell_l1_adjustment: float
    solve_seconds: float
    raw_target_reachable: bool


def _cell_objective(cells: np.ndarray, raw: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * np.abs(np.asarray(cells, dtype=float) - raw)))


def reconcile_reachable_cells_greedy(
    current_cells: np.ndarray,
    raw_target_cells: np.ndarray,
    marginals: np.ndarray,
) -> TemporalReconciliation:
    """Fast constructive projection onto the reachable cumulative-Venn set.

    Every marginal increment is implemented as a move from mask S to S plus
    one new EDP bit. Several deterministic EDP orders are tried, and the state
    with the smallest weighted cell error is retained. The result is integer,
    preserves all EDP reaches exactly, and is reachable by construction.
    """
    start = perf_counter()
    current = np.asarray(current_cells, dtype=int)
    raw = np.maximum(np.asarray(raw_target_cells, dtype=float), 0.0)
    n_edps = int(round(np.log2(len(current))))
    size = 1 << n_edps
    current_marginals = np.asarray(
        [sum(current[mask] for mask in range(size) if mask & (1 << edp)) for edp in range(n_edps)],
        dtype=int,
    )
    increments = np.asarray(marginals, dtype=int) - current_marginals
    if np.any(increments < 0):
        raise ValueError("cumulative EDP reaches cannot decrease")
    weights = 1.0 / np.sqrt(np.maximum(raw, 1.0))
    weights[0] = max(float(np.max(weights[raw > 0])), weights[0]) * 1_000.0

    descending = tuple(np.argsort(-increments).tolist())
    ascending = tuple(reversed(descending))
    orders = [descending, ascending]
    orders.extend(
        descending[offset:] + descending[:offset]
        for offset in range(0, n_edps, max(n_edps // 3, 1))
    )
    best_cells: np.ndarray | None = None
    best_value = float("inf")

    for order in dict.fromkeys(orders):
        cells = current.copy()
        for edp in order:
            remaining = int(increments[edp])
            bit = 1 << edp
            if remaining <= 0:
                continue
            segments: list[tuple[float, int, int, int]] = []
            for source in range(size):
                capacity = int(cells[source])
                if capacity <= 0 or source & bit:
                    continue
                destination = source | bit
                breakpoints = {0, capacity}
                for point in (
                    cells[source] - raw[source],
                    raw[destination] - cells[destination],
                ):
                    center = int(np.floor(point))
                    for candidate in (center - 1, center, center + 1, center + 2):
                        breakpoints.add(int(np.clip(candidate, 0, capacity)))
                ordered_points = sorted(breakpoints)
                for start_point, end_point in zip(ordered_points, ordered_points[1:]):
                    if end_point <= start_point:
                        continue
                    before = (
                        weights[source]
                        * abs(cells[source] - start_point - raw[source])
                        + weights[destination]
                        * abs(cells[destination] + start_point - raw[destination])
                    )
                    after = (
                        weights[source]
                        * abs(cells[source] - end_point - raw[source])
                        + weights[destination]
                        * abs(cells[destination] + end_point - raw[destination])
                    )
                    length = end_point - start_point
                    segments.append(
                        ((after - before) / length, source, destination, length)
                    )
            segments.sort()
            allocations: dict[tuple[int, int], int] = {}
            for _, source, destination, capacity in segments:
                if remaining <= 0:
                    break
                amount = min(remaining, capacity)
                allocations[(source, destination)] = (
                    allocations.get((source, destination), 0) + amount
                )
                remaining -= amount
            if remaining:
                raise RuntimeError("greedy temporal reconciliation found insufficient capacity")
            for (source, destination), amount in allocations.items():
                cells[source] -= amount
                cells[destination] += amount
        value = _cell_objective(cells, raw, weights)
        if value < best_value:
            best_value = value
            best_cells = cells

    if best_cells is None:
        raise RuntimeError("greedy temporal reconciliation produced no candidate")
    target = best_cells
    final_marginals = np.asarray(
        [sum(target[mask] for mask in range(size) if mask & (1 << edp)) for edp in range(n_edps)],
        dtype=int,
    )
    if not np.array_equal(final_marginals, np.asarray(marginals, dtype=int)):
        raise RuntimeError("greedy temporal reconciliation changed an EDP marginal")
    raw_union = float(raw[1:].sum())
    reconciled_union = int(target[1:].sum())
    return TemporalReconciliation(
        target_cells=target,
        raw_union=raw_union,
        reconciled_union=reconciled_union,
        union_adjustment=abs(reconciled_union - raw_union) / max(raw_union, 1.0),
        cell_l1_adjustment=float(np.sum(np.abs(target - raw))) / max(2.0 * raw_union, 1.0),
        solve_seconds=perf_counter() - start,
        raw_target_reachable=False,
    )


@lru_cache(maxsize=None)
def _temporal_milp_structure(n_edps: int):
    size = 1 << n_edps
    sources: list[int] = []
    destinations: list[int] = []
    for source in range(size):
        remaining = (size - 1) ^ source
        submask = remaining
        while True:
            sources.append(source)
            destinations.append(source | submask)
            if submask == 0:
                break
            submask = (submask - 1) & remaining
    source_array = np.asarray(sources, dtype=np.int64)
    destination_array = np.asarray(destinations, dtype=np.int64)
    edge_count = len(source_array)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    # Every person in a prior exact cell must leave that source once.
    edge_columns = np.arange(edge_count, dtype=np.int64)
    rows.extend(source_array.tolist())
    columns.extend(edge_columns.tolist())
    values.extend(np.ones(edge_count).tolist())

    # Destination totals define the reconciled Venn cells.
    destination_offset = size
    rows.extend((destination_offset + destination_array).tolist())
    columns.extend(edge_columns.tolist())
    values.extend(np.ones(edge_count).tolist())
    rows.extend((destination_offset + np.arange(size)).tolist())
    columns.extend((edge_count + np.arange(size)).tolist())
    values.extend((-np.ones(size)).tolist())

    # Preserve the directly measured cumulative reach at every EDP.
    marginal_offset = 2 * size
    for edp in range(n_edps):
        masks = np.asarray([mask for mask in range(size) if mask & (1 << edp)], dtype=int)
        rows.extend(np.full(len(masks), marginal_offset + edp).tolist())
        columns.extend((edge_count + masks).tolist())
        values.extend(np.ones(len(masks)).tolist())

    # L1 distance between the reconciled cells and the raw calibrated cells.
    deviation_offset = marginal_offset + n_edps
    indices = np.arange(size, dtype=np.int64)
    rows.extend((deviation_offset + indices).tolist())
    columns.extend((edge_count + indices).tolist())
    values.extend(np.ones(size).tolist())
    rows.extend((deviation_offset + indices).tolist())
    columns.extend((edge_count + size + indices).tolist())
    values.extend((-np.ones(size)).tolist())
    rows.extend((deviation_offset + indices).tolist())
    columns.extend((edge_count + 2 * size + indices).tolist())
    values.extend(np.ones(size).tolist())

    matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(deviation_offset + size, edge_count + 3 * size),
    ).tocsr()
    integrality = np.zeros(edge_count + 3 * size, dtype=int)
    integrality[edge_count : edge_count + size] = 1
    return source_array, destination_array, matrix, integrality


def _integerize_static_cells(raw_cells: np.ndarray, marginals: np.ndarray) -> np.ndarray:
    """Round one static Venn table while preserving population and EDP reaches."""
    raw = np.asarray(raw_cells, dtype=float)
    size = len(raw)
    n_edps = int(round(np.log2(size)))
    floors = np.floor(raw + 1e-9).astype(int)
    fractions = raw - floors
    constraints = np.zeros((n_edps + 1, size), dtype=float)
    constraints[0] = 1.0
    for edp in range(n_edps):
        constraints[edp + 1] = [bool(mask & (1 << edp)) for mask in range(size)]
    target = np.r_[int(round(raw.sum())), np.asarray(marginals, dtype=int)]
    deficit = target - constraints @ floors
    result = milp(
        1.0 - 2.0 * fractions,
        integrality=np.ones(size, dtype=int),
        bounds=Bounds(np.zeros(size), np.ones(size)),
        constraints=LinearConstraint(sparse.csr_matrix(constraints), deficit, deficit),
        options={"time_limit": 30.0},
    )
    if not result.success:
        raise RuntimeError(f"static Venn integerization failed: {result.message}")
    rounded = floors + np.rint(result.x).astype(int)
    if np.max(np.abs(constraints @ rounded - target)) > 1e-6:
        raise RuntimeError("static Venn integerization did not preserve the marginals")
    return rounded


def reconcile_reachable_cells(
    current_cells: np.ndarray,
    raw_target_cells: np.ndarray,
    marginals: np.ndarray,
) -> TemporalReconciliation:
    """Find the closest integer Venn target reachable from frozen prior labels."""
    current = np.asarray(current_cells, dtype=int)
    raw = np.maximum(np.asarray(raw_target_cells, dtype=float), 0.0)
    n_edps = int(round(np.log2(len(current))))
    size = 1 << n_edps
    sources, destinations, matrix, integrality = _temporal_milp_structure(n_edps)
    edge_count = len(sources)
    population = int(current.sum())

    rhs = np.r_[
        current,
        np.zeros(size),
        np.asarray(marginals, dtype=float),
        raw,
    ]
    weights = 1.0 / np.sqrt(np.maximum(raw, 1.0))
    # Cell zero is the complement of union reach. Give it explicit weight so
    # temporal reconciliation does not obtain a good cell fit by moving the
    # headline union unnecessarily.
    weights[0] = max(float(np.max(weights[raw > 0])), weights[0]) * 1_000.0
    objective = np.r_[
        np.zeros(edge_count + size),
        weights,
        weights,
    ]
    lower = np.zeros(edge_count + 3 * size)
    upper = np.full(edge_count + 3 * size, np.inf)
    upper[edge_count : edge_count + size] = population

    start = perf_counter()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, rhs, rhs),
        options={"time_limit": 60.0, "mip_rel_gap": 1e-4},
    )
    elapsed = perf_counter() - start
    if not result.success:
        raise RuntimeError(f"temporal Venn reconciliation failed: {result.message}")
    target = np.rint(result.x[edge_count : edge_count + size]).astype(int)
    raw_reachable = bool(np.sum(np.abs(target - raw)) < 1e-6)
    raw_union = float(raw[1:].sum())
    reconciled_union = int(target[1:].sum())
    return TemporalReconciliation(
        target_cells=target,
        raw_union=raw_union,
        reconciled_union=reconciled_union,
        union_adjustment=abs(reconciled_union - raw_union) / max(raw_union, 1.0),
        cell_l1_adjustment=float(np.sum(np.abs(target - raw))) / max(2.0 * raw_union, 1.0),
        solve_seconds=elapsed,
        raw_target_reachable=raw_reachable,
    )


def label_from_cumulative_targets(
    campaign: Campaign,
    targets: list[np.ndarray],
    method: str,
    timing_policy: str = "active_today",
    flows: list[dict[tuple[int, int], int]] | None = None,
) -> LabelingResult:
    """Allocate immutable VIDs that exactly realize a reachable Venn sequence."""
    n_edps, n_weeks, n_users = campaign.events.shape
    labels = np.full(campaign.events.shape, -1, dtype=np.int64)
    memo: dict[int, int] = {}
    slot_members: dict[int, int] = {}
    slot_first_day: dict[int, int] = {}
    slot_last_seen: dict[int, int] = {}
    next_vid = 0
    all_edps = tuple(range(n_edps))
    current = np.zeros(1 << n_edps, dtype=int)
    current[0] = n_users

    for day, target in enumerate(targets):
        day_entries: list[tuple[int, np.ndarray, np.ndarray]] = []
        new_keys: dict[int, list[int]] = {edp: [] for edp in all_edps}
        for edp in all_edps:
            users = np.flatnonzero(campaign.events[edp, day])
            keys = ((edp + 1) * n_users + users + 1).astype(np.uint64)
            day_entries.append((edp, users, keys))
            new_keys[edp] = sorted(
                [int(key) for key in keys.tolist() if int(key) not in memo],
                key=lambda key: int(_mix64(key, 0xCA11B)[()]),
            )

        active_today: set[int] = set()
        for _, _, keys in day_entries:
            for key in keys.tolist():
                if int(key) in memo:
                    active_today.add(memo[int(key)])

        flow = flows[day] if flows is not None else _transport_cells(current, target)
        slots_by_mask: dict[int, list[int]] = {}
        for vid, mask in slot_members.items():
            slots_by_mask.setdefault(mask, []).append(vid)
        for values in slots_by_mask.values():
            if timing_policy == "recent_creation":
                values.sort(key=lambda vid: (slot_first_day[vid], vid), reverse=True)
            elif timing_policy == "oldest_creation":
                values.sort(key=lambda vid: (slot_first_day[vid], vid))
            elif timing_policy == "active_today":
                values.sort(
                    key=lambda vid: (
                        vid in active_today,
                        slot_last_seen.get(vid, slot_first_day[vid]),
                        slot_first_day[vid],
                        -vid,
                    ),
                    reverse=True,
                )
            else:
                raise ValueError(f"unknown timing policy: {timing_policy}")

        for (source, destination), count in sorted(flow.items()):
            if source == 0:
                selected_vids = list(range(next_vid, next_vid + count))
                next_vid += count
                for vid in selected_vids:
                    slot_first_day[vid] = day
                    slot_last_seen[vid] = day
            else:
                selected_vids = slots_by_mask[source][:count]
                del slots_by_mask[source][:count]
            additions = [
                edp
                for edp in all_edps
                if destination & (1 << edp) and not source & (1 << edp)
            ]
            for vid in selected_vids:
                for edp in additions:
                    if not new_keys[edp]:
                        raise RuntimeError("target flow used more new EDP identifiers than observed")
                    memo[new_keys[edp].pop()] = vid
                slot_members[vid] = destination
        if any(new_keys.values()):
            raise RuntimeError("target flow left new EDP identifiers unassigned")
        for edp, users, keys in day_entries:
            assigned = np.asarray(
                [memo[int(key)] for key in keys],
                dtype=np.int64,
            )
            labels[edp, day, users] = assigned
            for vid in assigned.tolist():
                slot_last_seen[int(vid)] = day
        current = np.asarray(target, dtype=int)

    return LabelingResult(
        f"{method}__{timing_policy}",
        labels,
        np.zeros(n_weeks),
        0,
        f"Provider-calibrated cumulative Venn targets; timing policy={timing_policy}.",
        supported_edps=n_edps,
        state_entries=len(memo),
        pool_count=(1 << n_edps) - 1,
        requires_ordered_days=True,
    )


def _calibration_specs(n_weeks: int, n_edps: int):
    all_edps = tuple(range(n_edps))
    return tuple(
        (f"weeks_1_{day + 1}", tuple(range(day + 1)), all_edps)
        for day in sorted({1, 3, 6, 9, n_weeks - 1})
    )


def _training_campaigns(world: SyntheticWorld, copies: int = 1) -> list[Campaign]:
    return [
        generate_campaign(
            world,
            scenario,
            world.config.seed + 10_000 + scenario_index * 100 + copy,
            f"provider_train_{scenario}_{copy}",
        )
        for scenario_index, scenario in enumerate(TRAINING_SCENARIOS)
        for copy in range(copies)
    ]


def _evaluation_campaigns(world: SyntheticWorld) -> list[Campaign]:
    campaigns = [
        generate_campaign(
            world,
            scenario,
            world.config.seed + 50_000 + index * 100,
            f"provider_eval_{scenario}",
        )
        for index, scenario in enumerate(EVALUATION_SCENARIOS)
    ]
    campaigns.extend(
        generate_temporal_stress_campaign(
            world,
            scenario,
            world.config.seed + 90_000 + index,
            f"provider_eval_{scenario}",
        )
        for index, scenario in enumerate(
            ("staggered_retargeting", "synchronized_retargeting", "shared_seed_then_expansion")
        )
    )
    return campaigns


def _full_population_reference_models(
    world: SyntheticWorld,
    campaigns: list[Campaign],
) -> dict[str, CalibrationModel]:
    specs = _calibration_specs(world.config.n_weeks, world.config.n_edps)
    observations = []
    for campaign in campaigns:
        for _, weeks, edps in specs:
            observation = measure_report(world, campaign, weeks, edps)
            observations.append(
                replace(
                    observation,
                    baseline_intersections=observation.truth_intersections.copy(),
                    baseline_unions=observation.truth_unions.copy(),
                )
            )
    dataset = calibration_dataset(observations, world.config.minimum_calibration_intersection)
    return {
        "fixed_log": PairAwareLogModel.fit(
            dataset,
            world.config.n_edps,
            "shared",
            world.config.ridge_penalty,
        ),
        "mixture": LatentMixtureModel.fit(
            dataset,
            world.config.n_edps,
            world.config.seed + 701,
        ),
    }


def _decoder_candidates(
    panel_models: dict[str, CalibrationModel],
    population_models: dict[str, CalibrationModel],
    evidence_half_saturation: float = 20.0,
) -> tuple[DecoderCandidate, ...]:
    def pairwise(model, name):
        return lambda observation: calibrate_report_pairwise_maximum_entropy(
            observation,
            model,
            pair_ridge=1e-6,
            evidence_half_saturation=evidence_half_saturation,
            name=name,
        )

    return (
        DecoderCandidate(
            "panel_fixed_log_pairwise",
            "5,000-person panel; fixed-plus-log capture; pairs followed by maximum-entropy closure.",
            pairwise(panel_models["fixed_log"], "panel_fixed_log_pairwise"),
        ),
        DecoderCandidate(
            "panel_direct_pairwise",
            "5,000-person panel; direct bounded fixed-plus-log pair capture; maximum-entropy closure.",
            pairwise(panel_models["direct_pair"], "panel_direct_pairwise"),
        ),
        DecoderCandidate(
            "panel_mixture_pairwise",
            "5,000-person panel; two matchability groups; pairs followed by maximum-entropy closure.",
            pairwise(panel_models["mixture"], "panel_mixture_pairwise"),
        ),
        DecoderCandidate(
            "panel_fixed_log_all_orders",
            "5,000-person panel; calibrated pair through ten-way inclusive intersections.",
            lambda observation: calibrate_report(observation, panel_models["fixed_log"]),
        ),
        DecoderCandidate(
            "panel_mixture_full_patterns",
            "5,000-person panel; all exact Reference-ID patterns decoded with two matchability groups.",
            lambda observation: calibrate_report_joint(
                observation,
                panel_models["mixture"],
                JointDecoderConfig(
                    "panel_mixture_full_patterns",
                    response_mode="mixture_exact",
                    prior_strength=1e-4,
                    evidence_half_saturation=5.0,
                ),
            ),
        ),
        DecoderCandidate(
            "population_mixture_pairwise",
            "Panel-noise upper bound using the full synthetic training population.",
            pairwise(population_models["mixture"], "population_mixture_pairwise"),
        ),
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _plot(rows: list[dict], output: Path) -> None:
    methods = [
        "existing_vid",
        "panel_fixed_log_pairwise__active_today",
        "panel_direct_pairwise__active_today",
        "panel_mixture_pairwise__active_today",
        "panel_fixed_log_all_orders__active_today",
        "panel_mixture_full_patterns__active_today",
        "population_mixture_pairwise__active_today",
    ]
    methods = [
        method
        for method in methods
        if any(row["metric"] == "report_error" and row["method"] == method for row in rows)
    ]
    labels = {
        "existing_vid": "Existing VID",
        "panel_fixed_log_pairwise__active_today": "Panel fixed+log / pairs",
        "panel_direct_pairwise__active_today": "Panel direct fixed+log / pairs",
        "panel_mixture_pairwise__active_today": "Panel mixture / pairs",
        "panel_fixed_log_all_orders__active_today": "Panel fixed+log / all orders",
        "panel_mixture_full_patterns__active_today": "Panel mixture / full patterns",
        "population_mixture_pairwise__active_today": "Full-training-population mixture / pairs",
    }
    data = [
        [
            100.0 * row["value"]
            for row in rows
            if row["metric"] == "report_error" and row["method"] == method
        ]
        for method in methods
    ]
    fig, axis = plt.subplots(figsize=(10, 5.8))
    axis.boxplot(
        data,
        tick_labels=[labels.get(method, method) for method in methods],
        vert=False,
        showfliers=False,
    )
    axis.set_xlabel("Absolute union-reach error (%)")
    axis.set_title("End-to-end calibrated Venn labeling")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_calibrated_venn_labeling(
    output_dir: Path,
    profile: str = "quick",
    candidate_names: set[str] | None = None,
    evidence_half_saturation: float = 20.0,
) -> dict:
    if profile not in {"smoke", "quick", "full"}:
        raise ValueError(f"unknown profile: {profile}")
    config = SimulationConfig(
        n_users=12_000 if profile == "full" else 4_000 if profile == "quick" else 1_000,
        population_size=180_000_000,
        n_edps=10,
        n_weeks=13,
        panel_size=5_000,
        minimum_calibration_intersection=30_000.0,
        ridge_penalty=0.35,
        seed=20260831,
    )
    world = make_world(config)
    training = _training_campaigns(world, copies=2 if profile == "full" else 1)
    evaluation = _evaluation_campaigns(world)
    if profile == "smoke":
        evaluation = evaluation[:2]
    specs = _calibration_specs(config.n_weeks, config.n_edps)
    panel = draw_panel(world, "representative", config.seed + 400_000, panel_size=5_000)
    panel_observations = list(_panel_observations(world, training, specs, panel).values())
    panel_models = _fit_reference_models(config, panel_observations, seed_offset=811)
    panel_truth_observations = [
        replace(
            observation,
            baseline_intersections=observation.truth_intersections.copy(),
            baseline_unions=observation.truth_unions.copy(),
        )
        for observation in panel_observations
    ]
    panel_models["direct_pair"] = DirectPairLogModel.fit(
        calibration_dataset(
            panel_truth_observations,
            config.minimum_calibration_intersection,
        ),
        config.n_edps,
        config.ridge_penalty,
    )
    population_models = _full_population_reference_models(world, training)
    candidates = _decoder_candidates(
        panel_models,
        population_models,
        evidence_half_saturation=evidence_half_saturation,
    )
    if profile == "smoke":
        candidates = candidates[:2]
    if candidate_names is not None:
        candidates = tuple(
            candidate for candidate in candidates if candidate.name in candidate_names
        )
        missing = candidate_names - {candidate.name for candidate in candidates}
        if missing:
            raise ValueError(f"unknown candidate names: {sorted(missing)}")
    all_edps = tuple(range(config.n_edps))
    rows: list[dict] = []
    candidate_metadata = {
        candidate.name: candidate.description for candidate in candidates
    }

    for campaign in evaluation:
        observations = [
            measure_report(world, campaign, tuple(range(day + 1)), all_edps)
            for day in range(config.n_weeks)
        ]
        for report_name, weeks, edps in _proof_report_specs(
            config.n_edps,
            config.n_weeks,
        ):
            observation = measure_report(world, campaign, weeks, edps)
            rows.append(
                {
                    "campaign_id": campaign.campaign_id,
                    "scenario": campaign.scenario,
                    "method": "existing_vid",
                    "metric": "report_error",
                    "report": report_name,
                    "report_type": (
                        "prefix"
                        if weeks == tuple(range(max(weeks) + 1))
                        else "interval"
                        if weeks == tuple(range(min(weeks), max(weeks) + 1))
                        else "noncontiguous"
                    ),
                    "edp_count": len(edps),
                    "day": max(weeks) + 1,
                    "value": abs(float(observation.baseline_unions[-1]) - float(observation.truth_unions[-1]))
                    / max(float(observation.truth_unions[-1]), 1.0),
                }
            )

        for candidate in candidates:
            current = np.zeros(1 << config.n_edps, dtype=int)
            current[0] = config.n_users
            targets: list[np.ndarray] = []
            for day, observation in enumerate(observations):
                decoded = candidate.decode(observation)
                raw_cells = decoded.exclusive_cells / observation.person_weight
                marginals = np.asarray(
                    [
                        int(round(observation.truth_intersections[1 << edp] / observation.person_weight))
                        for edp in range(config.n_edps)
                    ],
                    dtype=int,
                )
                reconciled = reconcile_reachable_cells_greedy(
                    current,
                    raw_cells,
                    marginals,
                )
                targets.append(reconciled.target_cells)
                rows.extend(
                    [
                        {
                            "campaign_id": campaign.campaign_id,
                            "scenario": campaign.scenario,
                            "method": candidate.name,
                            "metric": "raw_prefix_error",
                            "report": f"weeks_1_{day + 1}__10_edps",
                            "report_type": "prefix",
                            "edp_count": 10,
                            "day": day + 1,
                            "value": abs(decoded.full_union - observation.truth_unions[-1])
                            / max(observation.truth_unions[-1], 1.0),
                        },
                        {
                            "campaign_id": campaign.campaign_id,
                            "scenario": campaign.scenario,
                            "method": candidate.name,
                            "metric": "reconciled_prefix_error",
                            "report": f"weeks_1_{day + 1}__10_edps",
                            "report_type": "prefix",
                            "edp_count": 10,
                            "day": day + 1,
                            "value": abs(
                                reconciled.reconciled_union
                                - observation.truth_unions[-1] / observation.person_weight
                            )
                            / max(observation.truth_unions[-1] / observation.person_weight, 1.0),
                        },
                        {
                            "campaign_id": campaign.campaign_id,
                            "scenario": campaign.scenario,
                            "method": candidate.name,
                            "metric": "union_adjustment",
                            "report": f"weeks_1_{day + 1}__10_edps",
                            "report_type": "prefix",
                            "edp_count": 10,
                            "day": day + 1,
                            "value": reconciled.union_adjustment,
                        },
                        {
                            "campaign_id": campaign.campaign_id,
                            "scenario": campaign.scenario,
                            "method": candidate.name,
                            "metric": "cell_l1_adjustment",
                            "report": f"weeks_1_{day + 1}__10_edps",
                            "report_type": "prefix",
                            "edp_count": 10,
                            "day": day + 1,
                            "value": reconciled.cell_l1_adjustment,
                        },
                        {
                            "campaign_id": campaign.campaign_id,
                            "scenario": campaign.scenario,
                            "method": candidate.name,
                            "metric": "temporal_projection_used",
                            "report": f"weeks_1_{day + 1}__10_edps",
                            "report_type": "prefix",
                            "edp_count": 10,
                            "day": day + 1,
                            "value": float(reconciled.cell_l1_adjustment > 1e-6),
                        },
                        {
                            "campaign_id": campaign.campaign_id,
                            "scenario": campaign.scenario,
                            "method": candidate.name,
                            "metric": "reconciliation_seconds",
                            "report": f"weeks_1_{day + 1}__10_edps",
                            "report_type": "prefix",
                            "edp_count": 10,
                            "day": day + 1,
                            "value": reconciled.solve_seconds,
                        },
                    ]
                )
                current = reconciled.target_cells

            target_flows: list[dict[tuple[int, int], int]] = []
            previous = np.zeros(1 << config.n_edps, dtype=int)
            previous[0] = config.n_users
            for target in targets:
                target_flows.append(_transport_cells(previous, target))
                previous = target

            for timing_policy in (
                "active_today",
                "recent_creation",
                "oldest_creation",
            ):
                labeled = label_from_cumulative_targets(
                    campaign,
                    targets,
                    candidate.name,
                    timing_policy=timing_policy,
                    flows=target_flows,
                )
                for report_name, weeks, edps in _proof_report_specs(
                    config.n_edps,
                    config.n_weeks,
                ):
                    truth = _truth_union(campaign, weeks, edps)
                    estimate = _report_union(labeled.labels, weeks, edps)
                    rows.append(
                        {
                            "campaign_id": campaign.campaign_id,
                            "scenario": campaign.scenario,
                            "method": labeled.method,
                            "metric": "report_error",
                            "report": report_name,
                            "report_type": (
                                "prefix"
                                if weeks == tuple(range(max(weeks) + 1))
                                else "interval"
                                if weeks == tuple(range(min(weeks), max(weeks) + 1))
                                else "noncontiguous"
                            ),
                            "edp_count": len(edps),
                            "day": max(weeks) + 1,
                            "value": abs(estimate - truth) / max(truth, 1),
                        }
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "calibrated_venn_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict = {
        "configuration": {
            "profile": profile,
            "n_users": config.n_users,
            "n_edps": config.n_edps,
            "n_weeks": config.n_weeks,
            "panel_size": min(config.panel_size, config.n_users),
            "panel_effective_size": panel.effective_size,
            "training_campaigns": len(training),
            "evaluation_campaigns": len(evaluation),
            "evidence_half_saturation": evidence_half_saturation,
        },
        "candidate_descriptions": candidate_metadata,
        "methods": {},
    }
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {
            metric: _summary([row["value"] for row in selected if row["metric"] == metric])
            for metric in sorted({row["metric"] for row in selected})
        }
        summary["methods"][method]["report_error_by_type"] = {
            report_type: _summary(
                [
                    row["value"]
                    for row in selected
                    if row["metric"] == "report_error" and row["report_type"] == report_type
                ]
            )
            for report_type in ("prefix", "interval", "noncontiguous")
            if any(
                row["metric"] == "report_error" and row["report_type"] == report_type
                for row in selected
            )
        }
    (output_dir / "calibrated_venn_summary.json").write_text(json.dumps(summary, indent=2))
    _plot(rows, output_dir / "calibrated_venn_error.png")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "quick", "full"), default="quick")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        nargs="*",
        help="Optional candidate names; omit to run the complete benchmark.",
    )
    parser.add_argument("--evidence-half-saturation", type=float, default=20.0)
    arguments = parser.parse_args()
    run_calibrated_venn_labeling(
        arguments.output_dir,
        arguments.profile,
        set(arguments.candidates) if arguments.candidates else None,
        arguments.evidence_half_saturation,
    )
