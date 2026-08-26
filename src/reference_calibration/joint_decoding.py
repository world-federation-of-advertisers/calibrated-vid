from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear, minimize
from scipy.special import logsumexp

from .evaluation import CalibratedReport
from .measurement import ReportObservation
from .models import CalibrationModel, LatentMixtureModel
from .research_models import MultiGroupMixtureModel
from .sets import enforce_marginals, intersection_matrix, members, union_values


@dataclass(frozen=True)
class JointDecoderConfig:
    name: str
    response_mode: str = "inclusive"
    prior_strength: float = 0.01
    evidence_half_saturation: float = 5.0
    map_iterations: int = 1


@lru_cache(maxsize=None)
def _pairwise_maximum_entropy_features(n_edps: int):
    pairs = tuple((left, right) for left in range(n_edps) for right in range(left + 1, n_edps))
    cells = np.arange(1 << n_edps, dtype=np.int64)
    features = np.zeros((len(cells), n_edps + len(pairs)), dtype=float)
    for cell_index, cell in enumerate(cells):
        for edp in range(n_edps):
            features[cell_index, edp] = bool(cell & (1 << edp))
        for pair_index, (left, right) in enumerate(pairs):
            features[cell_index, n_edps + pair_index] = bool(
                cell & (1 << left) and cell & (1 << right)
            )
    return features, pairs


def calibrate_report_pairwise_maximum_entropy(
    observation: ReportObservation,
    model: CalibrationModel,
    pair_ridge: float = 0.002,
    evidence_half_saturation: float = 30.0,
    name: str = "pairwise_maximum_entropy",
    pair_target_intersections: np.ndarray | None = None,
) -> CalibratedReport:
    """Infer all higher-order cells from calibrated singleton and pair moments."""
    n_edps = len(observation.edps)
    population = float(observation.truth_intersections[0])
    capture = np.ones(len(observation.global_masks), dtype=float)
    pair_local_masks: list[int] = []
    pair_global_masks: list[int] = []
    pair_scales: list[float] = []
    for local_mask in range(1, len(observation.global_masks)):
        if local_mask.bit_count() != 2:
            continue
        local_members = members(local_mask, n_edps)
        geometric_mean = float(
            np.exp(
                np.mean(
                    np.log(
                        np.maximum(observation.reach_fractions[list(local_members)], 1e-9)
                    )
                )
            )
        )
        pair_local_masks.append(local_mask)
        pair_global_masks.append(int(observation.global_masks[local_mask]))
        pair_scales.append(float(np.log(max(geometric_mean, 1e-9))))
    pair_predictions = model.predict_capture(
        np.asarray(pair_global_masks, dtype=np.int64),
        np.asarray(pair_scales, dtype=float),
    )
    for local_mask, value in zip(pair_local_masks, pair_predictions):
        capture[local_mask] = value
    features, pairs = _pairwise_maximum_entropy_features(n_edps)
    target = np.zeros(n_edps + len(pairs), dtype=float)
    target[:n_edps] = observation.reach_fractions

    for pair_index, (left, right) in enumerate(pairs):
        local_mask = (1 << left) | (1 << right)
        lower = max(
            float(observation.truth_intersections[1 << left])
            + float(observation.truth_intersections[1 << right])
            - population,
            0.0,
        )
        upper = min(
            float(observation.truth_intersections[1 << left]),
            float(observation.truth_intersections[1 << right]),
        )
        if pair_target_intersections is not None:
            estimate = float(np.clip(pair_target_intersections[local_mask], lower, upper))
            target[n_edps + pair_index] = estimate / population
        else:
            signal = max(float(observation.reference_signal[local_mask]), 0.0)
            estimate = signal / max(float(capture[local_mask]), 1e-9)
            estimate = float(np.clip(estimate, lower, upper))
            effective = signal / observation.person_weight
            reliability = effective / (effective + evidence_half_saturation)
            baseline = float(observation.baseline_intersections[local_mask])
            target[n_edps + pair_index] = (
                reliability * estimate + (1.0 - reliability) * baseline
            ) / population

    marginal_logits = np.log(
        np.clip(target[:n_edps], 1e-7, 1.0 - 1e-7)
        / np.clip(1.0 - target[:n_edps], 1e-7, 1.0)
    )
    initial = np.concatenate([marginal_logits, np.zeros(len(pairs))])
    penalty = np.concatenate(
        [np.full(n_edps, pair_ridge * 0.01), np.full(len(pairs), pair_ridge)]
    )

    def objective(theta: np.ndarray):
        scores = features @ theta
        log_partition = logsumexp(scores)
        probabilities = np.exp(scores - log_partition)
        moments = probabilities @ features
        value = log_partition - float(theta @ target) + 0.5 * float(np.sum(penalty * theta * theta))
        gradient = moments - target + penalty * theta
        return value, gradient

    result = minimize(
        lambda theta: objective(theta),
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1_000, "ftol": 1e-12, "gtol": 1e-9},
    )
    scores = features @ result.x
    probabilities = np.exp(scores - logsumexp(scores))
    probabilities = np.maximum(probabilities, 1e-12)
    probabilities /= probabilities.sum()
    cells = probabilities * population
    marginals = np.asarray(
        [observation.truth_intersections[1 << local] for local in range(n_edps)],
        dtype=float,
    )
    cells = enforce_marginals(
        cells,
        marginals,
        population,
        max_iterations=10_000,
        tolerance=1e-9,
    )
    fitted_moments = (cells / population) @ features
    residual = float(np.max(np.abs(fitted_moments - target)))
    intersections = np.zeros(1 << n_edps, dtype=float)
    intersections[1:] = intersection_matrix(n_edps) @ cells[1:]
    return CalibratedReport(
        model_name=name,
        observation=observation,
        capture_rates=capture,
        raw_intersections=intersections,
        exclusive_cells=cells,
        union_values=union_values(cells),
        decoder_residual=residual,
    )


def _capture_vector(observation: ReportObservation, model: CalibrationModel) -> np.ndarray:
    size = len(observation.global_masks)
    capture = np.ones(size, dtype=float)
    global_masks: list[int] = []
    scales: list[float] = []
    local_masks: list[int] = []
    for local_mask in range(1, size):
        if local_mask.bit_count() < 2:
            continue
        local_members = members(local_mask, len(observation.edps))
        geometric_mean = float(
            np.exp(
                np.mean(
                    np.log(
                        np.maximum(
                            observation.reach_fractions[list(local_members)],
                            1e-9,
                        )
                    )
                )
            )
        )
        global_masks.append(int(observation.global_masks[local_mask]))
        scales.append(float(np.log(max(geometric_mean, 1e-9))))
        local_masks.append(local_mask)
    prediction = model.predict_capture(
        np.asarray(global_masks, dtype=np.int64),
        np.asarray(scales, dtype=float),
    )
    for local_mask, value in zip(local_masks, prediction):
        capture[local_mask] = value
    return capture


def _baseline_cells(observation: ReportObservation) -> np.ndarray:
    size = len(observation.global_masks)
    n_edps = len(observation.edps)
    population = float(observation.truth_intersections[0])
    result = np.zeros(size, dtype=float)
    for cell in range(size):
        probability = 1.0
        for local in range(n_edps):
            probability *= (
                observation.reach_fractions[local]
                if cell & (1 << local)
                else 1.0 - observation.reach_fractions[local]
            )
        result[cell] = population * probability
    return result


def _exact_reference_patterns(reference_intersections: np.ndarray) -> np.ndarray:
    """Recover exact shared-ID patterns of order two or greater.

    Singleton Reference IDs are intentionally excluded because a proprietary
    fallback ID cannot be distinguished from an email-derived ID when it
    appears at only one EDP.
    """
    size = len(reference_intersections)
    n_edps = int(round(np.log2(size)))
    exact = np.zeros(size, dtype=float)
    for order in range(n_edps, 1, -1):
        for mask in range(1, size):
            if mask.bit_count() != order:
                continue
            supersets = [
                other
                for other in range(mask + 1, size)
                if other != mask and other & mask == mask and other.bit_count() >= 2
            ]
            exact[mask] = reference_intersections[mask] - sum(exact[other] for other in supersets)
    return np.maximum(exact, 0.0)


def _mixture_components(model: CalibrationModel):
    if isinstance(model, LatentMixtureModel):
        return (
            np.asarray([model.class_weight, 1.0 - model.class_weight], dtype=float),
            np.asarray([model.low_link, model.high_link], dtype=float),
        )
    if isinstance(model, MultiGroupMixtureModel):
        return model.mixture_components()
    raise TypeError("exact-pattern response requires a latent mixture model")


def _affinity_score(mask: int, matrix: np.ndarray) -> float:
    selected = members(mask, matrix.shape[0])
    if len(selected) < 2:
        return 0.0
    return float(sum(matrix[i, j] for i in selected for j in selected if i < j))


def mixture_response_matrix(
    observation: ReportObservation,
    model: CalibrationModel,
    affinity_matrix: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Map true exact reach cells to exact shared-Reference-ID patterns."""
    class_weights, global_links = _mixture_components(model)
    links = global_links[:, np.asarray(observation.edps, dtype=int)]
    n_edps = len(observation.edps)
    size = 1 << n_edps
    row_masks = np.asarray(
        [mask for mask in range(1, size) if mask.bit_count() >= 2],
        dtype=np.int64,
    )
    row_index = {int(mask): index for index, mask in enumerate(row_masks)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    local_affinity = None
    if affinity_matrix is not None:
        selected = np.asarray(observation.edps, dtype=int)
        local_affinity = affinity_matrix[np.ix_(selected, selected)]

    for true_cell in range(1, size):
        possible: list[int] = []
        probabilities: list[float] = []
        submask = true_cell
        while True:
            probability = 0.0
            for group, group_weight in enumerate(class_weights):
                group_probability = 1.0
                for edp in members(true_cell, n_edps):
                    group_probability *= (
                        links[group, edp]
                        if submask & (1 << edp)
                        else 1.0 - links[group, edp]
                    )
                probability += float(group_weight) * group_probability
            if local_affinity is not None and submask.bit_count() >= 2:
                probability *= float(np.exp(np.clip(_affinity_score(submask, local_affinity), -6.0, 6.0)))
            possible.append(submask)
            probabilities.append(probability)
            if submask == 0:
                break
            submask = (submask - 1) & true_cell

        probabilities_array = np.asarray(probabilities, dtype=float)
        total = float(probabilities_array.sum())
        if total <= 0:
            continue
        probabilities_array /= total
        for observed_mask, probability in zip(possible, probabilities_array):
            if observed_mask.bit_count() < 2 or probability <= 0:
                continue
            rows.append(row_index[observed_mask])
            columns.append(true_cell - 1)
            values.append(float(probability))

    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(row_masks), size - 1),
    ), row_masks


def _inclusive_response_matrix(
    observation: ReportObservation,
    capture: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    n_edps = len(observation.edps)
    row_masks = np.asarray(
        [mask for mask in range(1, 1 << n_edps) if mask.bit_count() >= 2],
        dtype=np.int64,
    )
    matrix = intersection_matrix(n_edps)[row_masks - 1]
    return sparse.diags(capture[row_masks]) @ matrix, row_masks


def _solve_response(
    observation: ReportObservation,
    response: sparse.csr_matrix,
    row_masks: np.ndarray,
    target_counts: np.ndarray,
    config: JointDecoderConfig,
) -> tuple[np.ndarray, float]:
    n_edps = len(observation.edps)
    population = float(observation.truth_intersections[0])
    baseline = _baseline_cells(observation)
    baseline_fraction = baseline[1:] / population
    target = np.maximum(target_counts, 0.0) / population
    effective = np.maximum(target_counts, 0.0) / observation.person_weight
    reliability = effective / (effective + config.evidence_half_saturation)
    simulated_population = population / observation.person_weight

    # Relative-error weighting, gated by the amount of actually observed
    # shared-ID evidence. A zero exact pattern contributes no hard zero.
    scale = np.maximum(target, 1.0 / simulated_population)
    reference_weights = np.sqrt(reliability) / scale
    reference_weights = np.clip(reference_weights, 0.0, 25_000.0)

    marginal_matrix = intersection_matrix(n_edps)[
        np.asarray([(1 << local) - 1 for local in range(n_edps)], dtype=int)
    ]
    marginal_target = observation.reach_fractions
    marginal_weights = 2_000.0 / np.maximum(marginal_target, 1e-5)

    matrices = [
        sparse.diags(reference_weights) @ response,
        sparse.diags(marginal_weights) @ marginal_matrix,
    ]
    targets = [reference_weights * target, marginal_weights * marginal_target]

    prior_strength = max(config.prior_strength, 0.0)
    if prior_strength > 0:
        if config.map_iterations > 1:
            prior_weights = np.sqrt(prior_strength) / np.sqrt(
                np.maximum(baseline_fraction, 1.0 / simulated_population)
            )
        else:
            prior_weights = np.full_like(baseline_fraction, np.sqrt(prior_strength))
        matrices.append(sparse.diags(prior_weights))
        targets.append(prior_weights * baseline_fraction)

    design = sparse.vstack(matrices, format="csr")
    rhs = np.concatenate(targets)
    solution = lsq_linear(
        design,
        rhs,
        bounds=(0.0, 1.0),
        lsmr_tol="auto",
        max_iter=250,
    )

    # Optional IRLS turns the first block into a Poisson-likelihood
    # approximation while retaining a Gaussian empirical-Bayes cell prior.
    x = np.maximum(solution.x, 0.0)
    for _ in range(max(config.map_iterations - 1, 0)):
        mean = np.maximum(response @ x, 1.0 / simulated_population)
        poisson_weights = np.sqrt(reliability) / np.sqrt(mean)
        poisson_weights = np.clip(poisson_weights, 0.0, 25_000.0)
        matrices[0] = sparse.diags(poisson_weights) @ response
        targets[0] = poisson_weights * target
        design = sparse.vstack(matrices, format="csr")
        rhs = np.concatenate(targets)
        solution = lsq_linear(
            design,
            rhs,
            bounds=(0.0, 1.0),
            lsmr_tol="auto",
            max_iter=250,
        )
        x = np.maximum(solution.x, 0.0)

    cells = np.zeros(1 << n_edps, dtype=float)
    cells[1:] = x * population
    cells[0] = max(population - cells[1:].sum(), population * 1e-15)
    marginals = np.asarray(
        [observation.truth_intersections[1 << local] for local in range(n_edps)],
        dtype=float,
    )
    cells = enforce_marginals(cells, marginals, population)

    predicted = response @ (cells[1:] / population)
    selected = reference_weights > 0
    denominator = max(float(np.linalg.norm(reference_weights[selected] * target[selected])), 1e-12)
    residual = float(
        np.linalg.norm(reference_weights[selected] * (predicted[selected] - target[selected]))
        / denominator
    ) if np.any(selected) else 0.0
    return cells, residual


def calibrate_report_joint(
    observation: ReportObservation,
    model: CalibrationModel,
    config: JointDecoderConfig,
    affinity_matrix: np.ndarray | None = None,
) -> CalibratedReport:
    capture = _capture_vector(observation, model)
    if config.response_mode == "mixture_exact":
        response, row_masks = mixture_response_matrix(observation, model, affinity_matrix)
        exact_reference = _exact_reference_patterns(observation.reference_signal)
        target_counts = exact_reference[row_masks]
    elif config.response_mode == "inclusive":
        response, row_masks = _inclusive_response_matrix(observation, capture)
        target_counts = observation.reference_signal[row_masks]
    else:
        raise ValueError(f"unknown response mode: {config.response_mode}")

    cells, residual = _solve_response(
        observation,
        response,
        row_masks,
        target_counts,
        config,
    )
    fitted_intersections = np.zeros(len(observation.global_masks), dtype=float)
    fitted_intersections[1:] = intersection_matrix(len(observation.edps)) @ cells[1:]
    return CalibratedReport(
        model_name=config.name,
        observation=observation,
        capture_rates=capture,
        raw_intersections=fitted_intersections,
        exclusive_cells=cells,
        union_values=union_values(cells),
        decoder_residual=residual,
    )
