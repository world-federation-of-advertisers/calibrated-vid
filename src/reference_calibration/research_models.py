from __future__ import annotations

from dataclasses import dataclass
import itertools
from math import comb

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from .measurement import CalibrationDataset
from .models import (
    CalibrationModel,
    _aggregate_by_subset,
    _balance_campaign_weights,
    _logit,
    _sigmoid,
)
from .sets import members


def _membership(masks: np.ndarray, n_edps: int) -> np.ndarray:
    return np.asarray(
        [[bool(int(mask) & (1 << edp)) for edp in range(n_edps)] for mask in masks],
        dtype=float,
    )


def _pair_membership(masks: np.ndarray, n_edps: int, normalize: bool = True) -> np.ndarray:
    pairs = tuple(itertools.combinations(range(n_edps), 2))
    result = np.asarray(
        [
            [bool(int(mask) & (1 << left) and int(mask) & (1 << right)) for left, right in pairs]
            for mask in masks
        ],
        dtype=float,
    )
    if normalize:
        result /= np.maximum(result.sum(axis=1, keepdims=True), 1.0)
    return result


def _training_arrays(dataset: CalibrationDataset, n_edps: int):
    response = _logit(dataset.signal / np.maximum(dataset.k0, 1.0))
    weights = dataset.weight / max(float(np.median(dataset.weight)), 1.0)
    weights = np.clip(weights, 0.15, 8.0)
    weights /= np.sqrt(
        np.asarray([comb(n_edps, int(order)) for order in dataset.subset_orders], dtype=float)
    )
    weights = _balance_campaign_weights(dataset.campaign_ids, weights)
    return response, weights


@dataclass(frozen=True)
class DirectPairLogModel(CalibrationModel):
    """Direct bounded c_ij = a_ij + b log(size), fitted on pair rows only."""

    n_edps: int
    pair_intercepts: np.ndarray
    scale_slope: float
    scale_mean: float
    ridge_penalty: float
    name: str = "direct_pair_fixed_plus_log"

    @property
    def parameter_count(self) -> int:
        return len(self.pair_intercepts) + 1

    @classmethod
    def fit(
        cls,
        dataset: CalibrationDataset,
        n_edps: int,
        ridge_penalty: float = 0.35,
    ) -> "DirectPairLogModel":
        selected = dataset.subset_orders == 2
        masks = dataset.subset_masks[selected]
        scale = dataset.log_scale[selected]
        scale_mean = float(np.mean(scale))
        centered_scale = scale - scale_mean
        pair_list = tuple(itertools.combinations(range(n_edps), 2))
        pair_index = {((1 << left) | (1 << right)): index for index, (left, right) in enumerate(pair_list)}
        design = np.zeros((len(masks), len(pair_list) + 1), dtype=float)
        for row, mask in enumerate(masks):
            design[row, pair_index[int(mask)]] = 1.0
            design[row, -1] = centered_scale[row]
        response = np.clip(
            dataset.signal[selected] / np.maximum(dataset.k0[selected], 1.0),
            0.0,
            1.0,
        )
        weights = dataset.weight[selected] / max(float(np.median(dataset.weight[selected])), 1.0)
        weights = np.clip(weights, 0.15, 8.0)
        weights = _balance_campaign_weights(dataset.campaign_ids[selected], weights)
        penalty = np.full(design.shape[1], ridge_penalty, dtype=float)
        penalty[:-1] *= 0.10
        penalty[-1] *= 0.05
        augmented_design = np.vstack(
            [design * weights[:, None], np.diag(np.sqrt(penalty))]
        )
        augmented_response = np.concatenate(
            [response * weights, np.zeros(design.shape[1])]
        )
        coefficients, *_ = np.linalg.lstsq(
            augmented_design,
            augmented_response,
            rcond=None,
        )
        return cls(
            n_edps=n_edps,
            pair_intercepts=coefficients[:-1],
            scale_slope=float(coefficients[-1]),
            scale_mean=scale_mean,
            ridge_penalty=ridge_penalty,
        )

    def predict_capture(self, subset_masks: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        pair_list = tuple(itertools.combinations(range(self.n_edps), 2))
        pair_index = {((1 << left) | (1 << right)): index for index, (left, right) in enumerate(pair_list)}
        result = np.zeros(len(subset_masks), dtype=float)
        for row, (mask, scale) in enumerate(zip(subset_masks, log_scale)):
            if int(mask).bit_count() != 2:
                raise ValueError("DirectPairLogModel supports pairwise capture only")
            result[row] = self.pair_intercepts[pair_index[int(mask)]] + self.scale_slope * (
                scale - self.scale_mean
            )
        return np.clip(result, 1e-5, 1.0 - 1e-5)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": "direct_pair_fixed_plus_log",
            "n_edps": self.n_edps,
            "parameter_count": self.parameter_count,
            "pair_intercepts": self.pair_intercepts.tolist(),
            "scale_slope": self.scale_slope,
            "scale_mean": self.scale_mean,
            "ridge_penalty": self.ridge_penalty,
        }


@dataclass(frozen=True)
class HierarchicalPairModel(CalibrationModel):
    """Order, EDP, and shrunk residual-pair effects plus campaign scale."""

    n_edps: int
    slope_mode: str
    coefficients: np.ndarray
    ridge_penalty: float
    name: str

    @property
    def parameter_count(self) -> int:
        return len(self.coefficients)

    @classmethod
    def fit(
        cls,
        dataset: CalibrationDataset,
        n_edps: int,
        slope_mode: str = "shared",
        ridge_penalty: float = 1.0,
    ) -> "HierarchicalPairModel":
        design, slices = _hierarchical_design(
            dataset.subset_masks,
            dataset.subset_orders,
            dataset.log_scale,
            n_edps,
            slope_mode,
        )
        response, weights = _training_arrays(dataset, n_edps)
        penalty = np.full(design.shape[1], ridge_penalty, dtype=float)
        penalty[slices["orders"]] *= 0.05
        penalty[slices["edps"]] *= 0.25
        penalty[slices["pairs"]] *= 1.0
        penalty[slices["slopes"]] *= 0.10
        augmented_design = np.vstack(
            [design * weights[:, None], np.diag(np.sqrt(penalty))]
        )
        augmented_response = np.concatenate(
            [response * weights, np.zeros(design.shape[1])]
        )
        coefficients, *_ = np.linalg.lstsq(
            augmented_design,
            augmented_response,
            rcond=None,
        )
        return cls(
            n_edps=n_edps,
            slope_mode=slope_mode,
            coefficients=coefficients,
            ridge_penalty=ridge_penalty,
            name=f"hierarchical_pair_{slope_mode}_ridge_{ridge_penalty:g}",
        )

    def predict_capture(self, subset_masks: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        orders = np.asarray([int(mask).bit_count() for mask in subset_masks], dtype=np.int8)
        design, _ = _hierarchical_design(
            subset_masks,
            orders,
            log_scale,
            self.n_edps,
            self.slope_mode,
        )
        return np.clip(_sigmoid(design @ self.coefficients), 1e-7, 1.0 - 1e-7)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": "hierarchical_pair_logit",
            "n_edps": self.n_edps,
            "slope_mode": self.slope_mode,
            "parameter_count": self.parameter_count,
            "ridge_penalty": self.ridge_penalty,
            "coefficients": self.coefficients.tolist(),
        }


def _hierarchical_design(
    subset_masks: np.ndarray,
    subset_orders: np.ndarray,
    log_scale: np.ndarray,
    n_edps: int,
    slope_mode: str,
):
    if slope_mode not in {"none", "shared", "by_order"}:
        raise ValueError(f"unknown slope_mode: {slope_mode}")
    order_columns = n_edps - 1
    edp_columns = n_edps - 1
    pair_columns = comb(n_edps, 2) - 1
    slope_columns = 0 if slope_mode == "none" else 1 if slope_mode == "shared" else n_edps - 1
    start_edps = order_columns
    start_pairs = start_edps + edp_columns
    start_slopes = start_pairs + pair_columns
    design = np.zeros(
        (len(subset_masks), order_columns + edp_columns + pair_columns + slope_columns),
        dtype=float,
    )
    edps = _membership(subset_masks, n_edps)
    pairs = _pair_membership(subset_masks, n_edps, normalize=True)
    for row, (order, scale) in enumerate(zip(subset_orders, log_scale)):
        design[row, int(order) - 2] = 1.0
        design[row, start_edps:start_pairs] = edps[row, :-1] - edps[row, -1]
        design[row, start_pairs:start_slopes] = pairs[row, :-1] - pairs[row, -1]
        if slope_mode == "shared":
            design[row, start_slopes] = scale
        elif slope_mode == "by_order":
            design[row, start_slopes + int(order) - 2] = scale
    return design, {
        "orders": slice(0, start_edps),
        "edps": slice(start_edps, start_pairs),
        "pairs": slice(start_pairs, start_slopes),
        "slopes": slice(start_slopes, design.shape[1]),
    }


@dataclass(frozen=True)
class MonotoneSplineCaptureModel(CalibrationModel):
    """Pair-aware logit model with a bounded monotone scale curve."""

    n_edps: int
    coefficients: np.ndarray
    scale_min: float
    scale_max: float
    knots: tuple[float, ...]
    direction: str
    ridge_penalty: float
    name: str

    @property
    def parameter_count(self) -> int:
        return len(self.coefficients)

    @classmethod
    def fit(
        cls,
        dataset: CalibrationDataset,
        n_edps: int,
        direction: str = "decreasing",
        ridge_penalty: float = 0.35,
        knots: tuple[float, ...] = (0.25, 0.50, 0.75),
    ) -> "MonotoneSplineCaptureModel":
        if direction not in {"increasing", "decreasing"}:
            raise ValueError("direction must be increasing or decreasing")
        scale_min = float(np.min(dataset.log_scale))
        scale_max = float(np.max(dataset.log_scale))
        design, spline_slice = _spline_design(
            dataset.subset_masks,
            dataset.subset_orders,
            dataset.log_scale,
            n_edps,
            scale_min,
            scale_max,
            knots,
        )
        response, weights = _training_arrays(dataset, n_edps)
        penalty = np.full(design.shape[1], ridge_penalty, dtype=float)
        penalty[: n_edps - 1] *= 0.10
        penalty[spline_slice] *= 0.20
        augmented_design = np.vstack(
            [design * weights[:, None], np.diag(np.sqrt(penalty))]
        )
        augmented_response = np.concatenate(
            [response * weights, np.zeros(design.shape[1])]
        )
        lower = np.full(design.shape[1], -np.inf)
        upper = np.full(design.shape[1], np.inf)
        if direction == "increasing":
            lower[spline_slice] = 0.0
        else:
            upper[spline_slice] = 0.0
        result = lsq_linear(
            augmented_design,
            augmented_response,
            bounds=(lower, upper),
            lsmr_tol="auto",
            max_iter=500,
        )
        return cls(
            n_edps=n_edps,
            coefficients=result.x,
            scale_min=scale_min,
            scale_max=scale_max,
            knots=knots,
            direction=direction,
            ridge_penalty=ridge_penalty,
            name=f"monotone_{direction}_spline",
        )

    def predict_capture(self, subset_masks: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        orders = np.asarray([int(mask).bit_count() for mask in subset_masks], dtype=np.int8)
        design, _ = _spline_design(
            subset_masks,
            orders,
            log_scale,
            self.n_edps,
            self.scale_min,
            self.scale_max,
            self.knots,
        )
        return np.clip(_sigmoid(design @ self.coefficients), 1e-7, 1.0 - 1e-7)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": "monotone_shape_constrained_spline",
            "n_edps": self.n_edps,
            "parameter_count": self.parameter_count,
            "direction": self.direction,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "knots": list(self.knots),
            "ridge_penalty": self.ridge_penalty,
            "coefficients": self.coefficients.tolist(),
        }


def _spline_design(
    subset_masks: np.ndarray,
    subset_orders: np.ndarray,
    log_scale: np.ndarray,
    n_edps: int,
    scale_min: float,
    scale_max: float,
    knots: tuple[float, ...],
):
    order_columns = n_edps - 1
    pair_columns = comb(n_edps, 2) - 1
    spline_columns = 1 + len(knots)
    start_pairs = order_columns
    start_spline = start_pairs + pair_columns
    design = np.zeros((len(subset_masks), start_spline + spline_columns), dtype=float)
    pair_values = _pair_membership(subset_masks, n_edps, normalize=True)
    denominator = max(scale_max - scale_min, 1e-8)
    normalized_scale = np.clip((log_scale - scale_min) / denominator, 0.0, 1.0)
    for row, (order, scale) in enumerate(zip(subset_orders, normalized_scale)):
        design[row, int(order) - 2] = 1.0
        design[row, start_pairs:start_spline] = pair_values[row, :-1] - pair_values[row, -1]
        design[row, start_spline] = scale
        for knot_index, knot in enumerate(knots):
            design[row, start_spline + 1 + knot_index] = max(scale - knot, 0.0)
    return design, slice(start_spline, design.shape[1])


@dataclass(frozen=True)
class LowRankAffinityModel(CalibrationModel):
    """Capture model with EDP main effects and low-rank pair affinities."""

    n_edps: int
    rank: int
    order_intercepts: np.ndarray
    edp_effects: np.ndarray
    embeddings: np.ndarray
    scale_slope: float
    scale_mean: float
    scale_std: float
    ridge_penalty: float
    objective_cost: float
    name: str

    @property
    def parameter_count(self) -> int:
        return (self.n_edps - 1) + self.n_edps + self.n_edps * self.rank + 1

    @classmethod
    def fit(
        cls,
        dataset: CalibrationDataset,
        n_edps: int,
        rank: int = 2,
        ridge_penalty: float = 0.05,
        seed: int = 0,
    ) -> "LowRankAffinityModel":
        response, weights = _training_arrays(dataset, n_edps)
        membership = _membership(dataset.subset_masks, n_edps)
        orders = dataset.subset_orders.astype(int)
        scale_mean = float(np.mean(dataset.log_scale))
        scale_std = max(float(np.std(dataset.log_scale)), 1e-8)
        scale = (dataset.log_scale - scale_mean) / scale_std

        order_init = np.array(
            [
                float(np.average(response[orders == order], weights=weights[orders == order]))
                for order in range(2, n_edps + 1)
            ]
        )

        def unpack(theta: np.ndarray):
            cursor = 0
            order_values = theta[cursor : cursor + n_edps - 1]
            cursor += n_edps - 1
            edp_raw = theta[cursor : cursor + n_edps]
            cursor += n_edps
            embedding = theta[cursor : cursor + n_edps * rank].reshape(n_edps, rank)
            cursor += n_edps * rank
            slope = float(theta[cursor])
            return order_values, edp_raw - edp_raw.mean(), embedding, slope

        def linear_prediction(theta: np.ndarray):
            order_values, edp_effects, embedding, slope = unpack(theta)
            result = order_values[orders - 2] + membership @ edp_effects + slope * scale
            summed = membership @ embedding
            squared_sum = np.sum(summed * summed, axis=1)
            sum_squared = membership @ np.sum(embedding * embedding, axis=1)
            pair_count = np.maximum(orders * (orders - 1) / 2.0, 1.0)
            result += 0.5 * (squared_sum - sum_squared) / pair_count
            return result

        def residual(theta: np.ndarray):
            data = weights * (linear_prediction(theta) - response)
            _, edp_effects, embedding, slope = unpack(theta)
            regularization = np.sqrt(ridge_penalty) * np.concatenate(
                [0.35 * edp_effects, embedding.ravel(), np.array([0.25 * slope])]
            )
            return np.concatenate([data, regularization])

        rng = np.random.default_rng(seed)
        best = None
        for _ in range(3):
            theta = np.concatenate(
                [
                    order_init + rng.normal(scale=0.05, size=n_edps - 1),
                    rng.normal(scale=0.05, size=n_edps),
                    rng.normal(scale=0.08, size=n_edps * rank),
                    np.array([-0.10 + rng.normal(scale=0.05)]),
                ]
            )
            result = least_squares(
                residual,
                theta,
                max_nfev=600,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            if best is None or result.cost < best.cost:
                best = result
        if best is None:
            raise RuntimeError("low-rank fit did not run")
        order_values, edp_effects, embedding, slope = unpack(best.x)
        return cls(
            n_edps=n_edps,
            rank=rank,
            order_intercepts=order_values,
            edp_effects=edp_effects,
            embeddings=embedding,
            scale_slope=slope,
            scale_mean=scale_mean,
            scale_std=scale_std,
            ridge_penalty=ridge_penalty,
            objective_cost=float(best.cost),
            name=f"low_rank_affinity_rank_{rank}",
        )

    def predict_capture(self, subset_masks: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        membership = _membership(subset_masks, self.n_edps)
        orders = np.asarray([int(mask).bit_count() for mask in subset_masks], dtype=int)
        scale = (log_scale - self.scale_mean) / self.scale_std
        result = self.order_intercepts[orders - 2] + membership @ self.edp_effects
        summed = membership @ self.embeddings
        squared_sum = np.sum(summed * summed, axis=1)
        sum_squared = membership @ np.sum(self.embeddings * self.embeddings, axis=1)
        pair_count = np.maximum(orders * (orders - 1) / 2.0, 1.0)
        result += 0.5 * (squared_sum - sum_squared) / pair_count
        result += self.scale_slope * scale
        return np.clip(_sigmoid(result), 1e-7, 1.0 - 1e-7)

    def pair_affinity_matrix(self) -> np.ndarray:
        matrix = self.embeddings @ self.embeddings.T
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": "low_rank_edp_affinity",
            "n_edps": self.n_edps,
            "rank": self.rank,
            "parameter_count": self.parameter_count,
            "order_intercepts": self.order_intercepts.tolist(),
            "edp_effects": self.edp_effects.tolist(),
            "embeddings": self.embeddings.tolist(),
            "scale_slope": self.scale_slope,
            "scale_mean": self.scale_mean,
            "scale_std": self.scale_std,
            "ridge_penalty": self.ridge_penalty,
            "objective_cost": self.objective_cost,
        }


@dataclass(frozen=True)
class MultiGroupMixtureModel(CalibrationModel):
    """Ordered latent matchability mixture with two or more groups."""

    n_edps: int
    n_groups: int
    class_weights: np.ndarray
    link_probabilities: np.ndarray
    objective_cost: float
    name: str

    @property
    def parameter_count(self) -> int:
        return (self.n_groups - 1) + self.n_groups * self.n_edps

    @classmethod
    def fit(
        cls,
        dataset: CalibrationDataset,
        n_edps: int,
        n_groups: int = 3,
        seed: int = 0,
    ) -> "MultiGroupMixtureModel":
        if n_groups < 2:
            raise ValueError("n_groups must be at least two")
        masks, target, weight = _aggregate_by_subset(dataset)
        membership = _membership(masks, n_edps)
        normalized_weight = np.sqrt(weight / max(float(np.median(weight)), 1.0))
        normalized_weight = np.clip(normalized_weight, 0.25, 12.0)
        normalized_weight /= np.sqrt(
            np.asarray([comb(n_edps, int(mask).bit_count()) for mask in masks], dtype=float)
        )

        def unpack(theta: np.ndarray):
            weight_logits = np.concatenate([theta[: n_groups - 1], np.zeros(1)])
            weight_logits -= np.max(weight_logits)
            class_weights = np.exp(weight_logits)
            class_weights /= class_weights.sum()
            cursor = n_groups - 1
            base = _sigmoid(theta[cursor : cursor + n_edps])
            cursor += n_edps
            links = [base]
            for _ in range(1, n_groups):
                gap = _sigmoid(theta[cursor : cursor + n_edps])
                cursor += n_edps
                links.append(links[-1] + (1.0 - links[-1]) * gap)
            return class_weights, np.asarray(links)

        def prediction(theta: np.ndarray):
            class_weights, links = unpack(theta)
            output = np.zeros(len(masks), dtype=float)
            for group in range(n_groups):
                output += class_weights[group] * np.prod(
                    np.where(membership > 0, links[group][None, :], 1.0),
                    axis=1,
                )
            return np.clip(output, 1e-9, 1.0)

        def residual(theta: np.ndarray):
            data = normalized_weight * (
                np.log(prediction(theta)) - np.log(np.clip(target, 1e-9, 1.0))
            )
            return np.concatenate([data, 0.012 * theta])

        rng = np.random.default_rng(seed)
        best = None
        for _ in range(6):
            theta = np.concatenate(
                [
                    rng.normal(scale=0.30, size=n_groups - 1),
                    -1.4 + rng.normal(scale=0.25, size=n_edps),
                    *[
                        rng.normal(loc=-0.1 + 0.25 * group, scale=0.35, size=n_edps)
                        for group in range(1, n_groups)
                    ],
                ]
            )
            result = least_squares(
                residual,
                theta,
                max_nfev=2500,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            if best is None or result.cost < best.cost:
                best = result
        if best is None:
            raise RuntimeError("multi-group mixture fit did not run")
        class_weights, links = unpack(best.x)
        return cls(
            n_edps=n_edps,
            n_groups=n_groups,
            class_weights=class_weights,
            link_probabilities=links,
            objective_cost=float(best.cost),
            name=f"{n_groups}_group_latent_mixture",
        )

    def predict_capture(self, subset_masks: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        del log_scale
        membership = _membership(subset_masks, self.n_edps)
        result = np.zeros(len(subset_masks), dtype=float)
        for group in range(self.n_groups):
            result += self.class_weights[group] * np.prod(
                np.where(membership > 0, self.link_probabilities[group][None, :], 1.0),
                axis=1,
            )
        return np.clip(result, 1e-9, 1.0)

    def mixture_components(self):
        return self.class_weights, self.link_probabilities

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": "multi_group_latent_matchability",
            "n_edps": self.n_edps,
            "n_groups": self.n_groups,
            "parameter_count": self.parameter_count,
            "class_weights": self.class_weights.tolist(),
            "link_probabilities": self.link_probabilities.tolist(),
            "objective_cost": self.objective_cost,
        }


def fit_low_rank_pair_affinity(
    dataset: CalibrationDataset,
    base_model: CalibrationModel,
    n_edps: int,
    rank: int = 2,
    maximum_absolute_affinity: float = 1.5,
) -> np.ndarray:
    """Fit a low-rank matrix to residual pairwise log capture ratios.

    The base latent mixture supplies the coherent independent-within-class
    response. This matrix captures repeatable pair-specific excess or deficit
    matching that remains after the mixture.
    """
    selected = dataset.subset_orders == 2
    masks = dataset.subset_masks[selected]
    scale = dataset.log_scale[selected]
    observed = np.clip(
        dataset.signal[selected] / np.maximum(dataset.k0[selected], 1.0),
        1e-8,
        1.0,
    )
    predicted = np.clip(base_model.predict_capture(masks, scale), 1e-8, 1.0)
    weights = _balance_campaign_weights(
        dataset.campaign_ids[selected],
        dataset.weight[selected],
    )
    pair_values: dict[int, list[tuple[float, float]]] = {}
    for mask, actual, expected, weight in zip(masks, observed, predicted, weights):
        pair_values.setdefault(int(mask), []).append(
            (float(np.log(actual) - np.log(expected)), float(weight))
        )
    residual_matrix = np.zeros((n_edps, n_edps), dtype=float)
    for mask, values in pair_values.items():
        selected_edps = members(mask, n_edps)
        if len(selected_edps) != 2:
            continue
        residual = float(
            np.average(
                [value for value, _ in values],
                weights=[weight for _, weight in values],
            )
        )
        left, right = selected_edps
        residual_matrix[left, right] = residual_matrix[right, left] = residual

    eigenvalues, eigenvectors = np.linalg.eigh(residual_matrix)
    selected_components = np.argsort(np.abs(eigenvalues))[::-1][:rank]
    approximation = (
        eigenvectors[:, selected_components]
        @ np.diag(eigenvalues[selected_components])
        @ eigenvectors[:, selected_components].T
    )
    np.fill_diagonal(approximation, 0.0)
    return np.clip(
        approximation,
        -maximum_absolute_affinity,
        maximum_absolute_affinity,
    )
