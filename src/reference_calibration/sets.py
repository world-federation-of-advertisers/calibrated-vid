from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares, lsq_linear
from scipy.special import logsumexp


def members(mask: int, n: int) -> tuple[int, ...]:
    return tuple(i for i in range(n) if mask & (1 << i))


def mask_for(indices) -> int:
    mask = 0
    for index in indices:
        mask |= 1 << int(index)
    return mask


def exact_cells_from_membership(membership: np.ndarray, weight: float) -> np.ndarray:
    """Return exact nonempty Venn cells from an EDP x person Boolean matrix."""
    n = membership.shape[0]
    powers = (1 << np.arange(n, dtype=np.int64))[:, None]
    masks = np.sum(membership.astype(np.int64) * powers, axis=0)
    return np.bincount(masks, minlength=1 << n).astype(float) * weight


def inclusive_intersections(exact: np.ndarray) -> np.ndarray:
    """Convert exact cells to inclusive intersections for every subset."""
    n = int(round(np.log2(len(exact))))
    values = np.asarray(exact, dtype=float).copy()
    for bit in range(n):
        step = 1 << bit
        for mask in range(1 << n):
            if not mask & step:
                values[mask] += values[mask | step]
    values[0] = exact.sum()
    return values


def union_values(exact: np.ndarray) -> np.ndarray:
    """Return union reach for every EDP subset from exact Venn cells."""
    n = int(round(np.log2(len(exact))))
    contained = np.asarray(exact, dtype=float).copy()
    for bit in range(n):
        step = 1 << bit
        for mask in range(1 << n):
            if mask & step:
                contained[mask] += contained[mask ^ step]
    full = (1 << n) - 1
    total = exact.sum()
    result = np.zeros(1 << n, dtype=float)
    for subset in range(1, 1 << n):
        result[subset] = total - contained[full ^ subset]
    return result


@lru_cache(maxsize=None)
def intersection_matrix(n: int) -> sparse.csr_matrix:
    """Map nonempty exact cells to nonempty inclusive intersections."""
    rows: list[int] = []
    cols: list[int] = []
    for subset in range(1, 1 << n):
        for cell in range(1, 1 << n):
            if cell & subset == subset:
                rows.append(subset - 1)
                cols.append(cell - 1)
    data = np.ones(len(rows), dtype=float)
    size = (1 << n) - 1
    return sparse.csr_matrix((data, (rows, cols)), shape=(size, size))


def decode_nonnegative_cells(
    target_intersections: np.ndarray,
    observation_weights: np.ndarray | None = None,
    union_prior: float | None = None,
    union_weight: float = 0.0,
    cell_prior: np.ndarray | None = None,
    cell_prior_weight: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Fit one nonnegative Venn diagram to noisy inclusive intersections."""
    n = int(round(np.log2(len(target_intersections))))
    if len(target_intersections) != 1 << n:
        raise ValueError("target_intersections must have length 2**n")
    matrix = intersection_matrix(n)
    target = np.asarray(target_intersections[1:], dtype=float)
    if observation_weights is None:
        weights = np.ones_like(target)
    else:
        weights = np.asarray(observation_weights[1:], dtype=float)
    weighted_matrix = sparse.diags(weights) @ matrix
    weighted_target = weights * target
    if union_prior is not None and union_weight > 0:
        union_row = sparse.csr_matrix(np.full((1, matrix.shape[1]), union_weight, dtype=float))
        weighted_matrix = sparse.vstack([weighted_matrix, union_row], format="csr")
        weighted_target = np.concatenate([weighted_target, [union_weight * union_prior]])
    if cell_prior is not None and cell_prior_weight > 0:
        prior = np.asarray(cell_prior[1:], dtype=float)
        prior_matrix = sparse.eye(matrix.shape[1], format="csr") * cell_prior_weight
        weighted_matrix = sparse.vstack([weighted_matrix, prior_matrix], format="csr")
        weighted_target = np.concatenate([weighted_target, cell_prior_weight * prior])
    result = lsq_linear(
        weighted_matrix,
        weighted_target,
        bounds=(0.0, np.inf),
        lsmr_tol="auto",
        max_iter=150,
    )
    cells = np.zeros(1 << n, dtype=float)
    cells[1:] = np.maximum(result.x, 0.0)
    fitted = matrix @ cells[1:]
    denominator = max(float(np.linalg.norm(weights * target)), 1.0)
    residual = float(np.linalg.norm(weights * (fitted - target)) / denominator)
    return cells, residual


def enforce_marginals(
    exact: np.ndarray,
    marginals: np.ndarray,
    population: float,
    max_iterations: int = 2_000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Project nonnegative Venn cells onto fixed one-EDP reaches.

    Iterative proportional fitting preserves nonnegativity while making the
    full table sum to the population and each one-EDP marginal equal its
    supplied VID reach. A small positive floor gives every feasible cell
    support, including the unreached population in cell zero.
    """
    values = np.asarray(exact, dtype=float).copy()
    n = int(round(np.log2(len(values))))
    targets = np.asarray(marginals, dtype=float)
    if len(targets) != n:
        raise ValueError("marginals must contain one value per EDP")
    if population <= 0 or np.any(targets < 0) or np.any(targets > population):
        raise ValueError("marginals must be between zero and the population")

    floor = max(population * 1e-15, 1e-9)
    values = np.maximum(values, floor)
    values[0] = max(population - float(values[1:].sum()), floor)
    values *= population / float(values.sum())
    base_probabilities = values / population
    masks = np.arange(len(values), dtype=np.int64)

    for _ in range(max_iterations):
        for edp, target in enumerate(targets):
            included = (masks & (1 << edp)) != 0
            current_in = float(values[included].sum())
            current_out = float(values[~included].sum())
            if target <= 0:
                values[included] = 0.0
                values[~included] *= population / max(current_out, floor)
            elif target >= population:
                values[~included] = 0.0
                values[included] *= population / max(current_in, floor)
            else:
                values[included] *= target / max(current_in, floor)
                values[~included] *= (population - target) / max(current_out, floor)

        fitted = np.array(
            [values[(masks & (1 << edp)) != 0].sum() for edp in range(n)],
            dtype=float,
        )
        relative_error = np.max(np.abs(fitted - targets) / np.maximum(targets, 1.0))
        if relative_error <= tolerance:
            break
    else:
        # Very sparse ten-EDP tables can make iterative proportional fitting
        # converge too slowly at strict tolerances. Fall back to the equivalent
        # convex exponential-tilting problem, which retains full support and
        # enforces the same singleton marginals.
        features = (
            (masks[:, None] & (1 << np.arange(n, dtype=np.int64))) != 0
        ).astype(float)
        log_base = np.log(np.maximum(base_probabilities, 1e-300))
        target_probabilities = targets / population

        scales = np.maximum(target_probabilities, 1.0 / population)

        def probabilities_for(coefficients: np.ndarray) -> np.ndarray:
            scores = log_base + features @ coefficients
            return np.exp(scores - logsumexp(scores))

        def residuals(coefficients: np.ndarray) -> np.ndarray:
            probabilities = probabilities_for(coefficients)
            return (probabilities @ features - target_probabilities) / scales

        def jacobian(coefficients: np.ndarray) -> np.ndarray:
            probabilities = probabilities_for(coefficients)
            fitted = probabilities @ features
            covariance = (
                (features.T * probabilities) @ features - np.outer(fitted, fitted)
            )
            return covariance / scales[:, None]

        result = least_squares(
            residuals,
            np.zeros(n, dtype=float),
            jac=jacobian,
            max_nfev=5_000,
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
            x_scale="jac",
        )
        probabilities = probabilities_for(result.x)
        values = probabilities * population
        fitted = values @ features
        relative_error = np.max(
            np.abs(fitted - targets) / np.maximum(targets, 1.0)
        )
        if relative_error > max(tolerance, 1e-8):
            raise RuntimeError(
                "marginal projection did not converge "
                f"(maximum relative error {relative_error:.3g})"
            )

    return values


def selected_masks(n: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(mask for mask in range(1, 1 << n) if mask.bit_count() in sizes)
