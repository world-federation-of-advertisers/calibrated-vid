from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from .measurement import ReportObservation
from .models import CalibrationModel
from .sets import decode_nonnegative_cells, enforce_marginals, intersection_matrix, members, union_values


@dataclass(frozen=True)
class CalibratedReport:
    model_name: str
    observation: ReportObservation
    capture_rates: np.ndarray
    raw_intersections: np.ndarray
    exclusive_cells: np.ndarray
    union_values: np.ndarray
    decoder_residual: float

    @property
    def full_union(self) -> float:
        return float(self.union_values[-1])


def calibrate_report(observation: ReportObservation, model: CalibrationModel) -> CalibratedReport:
    size = len(observation.global_masks)
    local_n = len(observation.edps)
    capture = np.ones(size, dtype=float)
    log_scale = np.zeros(size, dtype=float)

    selected_global_masks: list[int] = []
    selected_log_scale: list[float] = []
    selected_local_masks: list[int] = []
    for local_mask in range(1, size):
        if local_mask.bit_count() < 2:
            continue
        local_members = members(local_mask, local_n)
        geometric_mean = float(
            np.exp(np.mean(np.log(np.maximum(observation.reach_fractions[list(local_members)], 1e-9))))
        )
        log_scale[local_mask] = np.log(max(geometric_mean, 1e-9))
        selected_global_masks.append(int(observation.global_masks[local_mask]))
        selected_log_scale.append(log_scale[local_mask])
        selected_local_masks.append(local_mask)

    predicted = model.predict_capture(
        np.asarray(selected_global_masks, dtype=np.int64),
        np.asarray(selected_log_scale, dtype=float),
    )
    for local_mask, value in zip(selected_local_masks, predicted):
        capture[local_mask] = value

    raw = np.zeros(size, dtype=float)
    weights = np.ones(size, dtype=float)
    pair_reliability: list[float] = []
    for local_mask in range(1, size):
        local_members = members(local_mask, local_n)
        if len(local_members) == 1:
            raw[local_mask] = observation.truth_intersections[local_mask]
            weights[local_mask] = 50.0 / np.sqrt(local_n)
            continue
        upper = min(
            observation.truth_intersections[1 << local_index]
            for local_index in local_members
        )
        estimate = observation.reference_signal[local_mask] / max(capture[local_mask], 1e-8)
        estimate = float(np.clip(estimate, 0.0, upper))
        # Higher-order cells can contain only a handful of sampled people even
        # when the real-population count looks large after weighting. Shrink
        # those weak cells toward the representative VID reference rather than
        # amplifying a zero or one-person Reference-ID observation.
        effective_matches = max(observation.reference_signal[local_mask], 0.0) / observation.person_weight
        reliability = effective_matches / (effective_matches + 30.0)
        if len(local_members) == 2:
            pair_reliability.append(float(reliability))
        raw[local_mask] = reliability * estimate + (1.0 - reliability) * observation.baseline_intersections[local_mask]
        signal_strength = np.sqrt(max(effective_matches, 0.0))
        correlated_count = comb(local_n, len(local_members))
        weights[local_mask] = float(
            np.clip(0.25 + reliability * signal_strength, 0.25, 8.0)
            / np.sqrt(max(correlated_count, 1))
        )

    baseline_cells = np.zeros(size, dtype=float)
    for cell in range(size):
        probability = 1.0
        for local in range(local_n):
            probability *= (
                observation.reach_fractions[local]
                if cell & (1 << local)
                else 1.0 - observation.reach_fractions[local]
            )
        baseline_cells[cell] = observation.truth_intersections[0] * probability

    evidence = float(np.mean(pair_reliability)) if pair_reliability else 0.0
    prior_weight = 0.10 + 0.80 * (1.0 - evidence)
    union_weight = 1.0 + 4.0 * (1.0 - evidence)
    cells, residual = decode_nonnegative_cells(
        raw,
        weights,
        union_prior=float(observation.baseline_unions[-1]),
        union_weight=union_weight,
        cell_prior=baseline_cells,
        cell_prior_weight=prior_weight,
    )
    population = float(observation.truth_intersections[0])
    if cells.sum() > population:
        cells, residual = decode_nonnegative_cells(
            raw,
            weights,
            union_prior=population,
            union_weight=100.0,
            cell_prior=baseline_cells,
            cell_prior_weight=prior_weight,
        )
    marginals = np.array(
        [observation.truth_intersections[1 << local] for local in range(local_n)],
        dtype=float,
    )
    cells = enforce_marginals(cells, marginals, population)
    fitted = intersection_matrix(local_n) @ cells[1:]
    target = raw[1:]
    denominator = max(float(np.linalg.norm(weights[1:] * target)), 1.0)
    residual = float(np.linalg.norm(weights[1:] * (fitted - target)) / denominator)
    return CalibratedReport(
        model_name=model.name,
        observation=observation,
        capture_rates=capture,
        raw_intersections=raw,
        exclusive_cells=cells,
        union_values=union_values(cells),
        decoder_residual=residual,
    )


def relative_error(estimate: float, truth: float) -> float:
    return abs(estimate - truth) / max(truth, 1.0)


def observable_capture_residual(observation: ReportObservation, model: CalibrationModel) -> float:
    """Signed observable diagnostic; positive means more matching than expected.

    This intentionally uses K0 rather than synthetic truth, so it demonstrates
    the real diagnostic's confounding between linkage shift and true-overlap
    shift.
    """
    masks: list[int] = []
    scales: list[float] = []
    observed: list[float] = []
    weights: list[float] = []
    for local_mask in range(1, len(observation.global_masks)):
        if local_mask.bit_count() != 2:
            continue
        local_members = members(local_mask, len(observation.edps))
        geometric_mean = float(
            np.exp(np.mean(np.log(np.maximum(observation.reach_fractions[list(local_members)], 1e-9))))
        )
        denominator = max(observation.baseline_intersections[local_mask], 1.0)
        ratio = max(observation.reference_signal[local_mask] / denominator, 1e-8)
        masks.append(int(observation.global_masks[local_mask]))
        scales.append(float(np.log(max(geometric_mean, 1e-9))))
        observed.append(ratio)
        weights.append(np.sqrt(max(observation.reference_intersections[local_mask], 1.0)))
    if not masks:
        return 0.0
    predicted = model.predict_capture(np.asarray(masks, dtype=np.int64), np.asarray(scales))
    residual = np.log(np.asarray(observed)) - np.log(np.maximum(predicted, 1e-8))
    return float(np.average(residual, weights=np.asarray(weights)))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }
