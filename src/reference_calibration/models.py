from __future__ import annotations

from dataclasses import dataclass
import itertools
from math import comb

import numpy as np
from scipy.optimize import least_squares

from .measurement import CalibrationDataset


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-5, 1.0 - 1e-5)
    return np.log(value / (1.0 - value))


def _balance_campaign_weights(campaign_ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Give each calibration campaign the same total squared fitting weight."""
    result = np.asarray(weights, dtype=float).copy()
    campaign_norms: dict[object, float] = {}
    for campaign_id in np.unique(campaign_ids):
        selected = campaign_ids == campaign_id
        campaign_norms[campaign_id] = float(np.linalg.norm(result[selected]))
    positive = [value for value in campaign_norms.values() if value > 0]
    target = float(np.median(positive)) if positive else 1.0
    for campaign_id, norm in campaign_norms.items():
        if norm > 0:
            result[campaign_ids == campaign_id] *= target / norm
    return result


class CalibrationModel:
    name: str

    def predict_capture(self, subset_masks: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


def calibration_model_from_dict(data: dict) -> "CalibrationModel":
    """Recreate a fitted model from a versioned JSON artifact."""
    family = data.get("family")
    if family == "pair_aware_fixed_plus_log":
        coefficients = np.asarray(data["coefficients"], dtype=float)
        slope_mode = str(data["slope_mode"])
        slope_columns = 0 if slope_mode == "none" else 1 if slope_mode == "shared" else int(data["n_edps"]) - 1
        expected = (int(data["n_edps"]) - 1) + (comb(int(data["n_edps"]), 2) - 1) + slope_columns
        if len(coefficients) != expected:
            raise ValueError("pair-aware coefficient count does not match n_edps and slope_mode")
        return PairAwareLogModel(
            n_edps=int(data["n_edps"]),
            slope_mode=slope_mode,
            coefficients=coefficients,
            ridge_penalty=float(data["ridge_penalty"]),
            name=str(data["name"]),
        )
    if family == "latent_matchability_mixture":
        return LatentMixtureModel(
            n_edps=int(data["n_edps"]),
            class_weight=float(data["class_weight"]),
            low_link=np.asarray(data["low_link"], dtype=float),
            high_link=np.asarray(data["high_link"], dtype=float),
            objective_cost=float(data["objective_cost"]),
            name=str(data.get("name", "two_group_latent_mixture")),
        )
    raise ValueError(f"unknown model family: {family}")


@dataclass(frozen=True)
class PairAwareLogModel(CalibrationModel):
    n_edps: int
    slope_mode: str
    coefficients: np.ndarray
    ridge_penalty: float
    name: str

    @property
    def pair_list(self) -> tuple[tuple[int, int], ...]:
        return tuple(itertools.combinations(range(self.n_edps), 2))

    @property
    def parameter_count(self) -> int:
        return len(self.coefficients)

    @classmethod
    def fit(
        cls,
        dataset: CalibrationDataset,
        n_edps: int,
        slope_mode: str,
        ridge_penalty: float,
    ) -> "PairAwareLogModel":
        if slope_mode not in {"none", "shared", "by_order"}:
            raise ValueError(f"unknown slope_mode: {slope_mode}")
        design = _pair_design(dataset.subset_masks, dataset.subset_orders, dataset.log_scale, n_edps, slope_mode)
        response = _logit(dataset.signal / np.maximum(dataset.k0, 1.0))
        weights = dataset.weight / max(float(np.median(dataset.weight)), 1.0)
        weights = np.clip(weights, 0.15, 8.0)
        weights /= np.sqrt(
            np.array([comb(n_edps, int(order)) for order in dataset.subset_orders], dtype=float)
        )
        weights = _balance_campaign_weights(dataset.campaign_ids, weights)
        weighted_design = design * weights[:, None]
        weighted_response = response * weights

        penalty = np.full(design.shape[1], ridge_penalty, dtype=float)
        penalty[: n_edps - 1] = ridge_penalty * 0.15  # order intercepts
        augmented_design = np.vstack([weighted_design, np.diag(np.sqrt(penalty))])
        augmented_response = np.concatenate([weighted_response, np.zeros(design.shape[1])])
        coefficients, *_ = np.linalg.lstsq(augmented_design, augmented_response, rcond=None)
        label = {
            "none": "pair_aware_fixed",
            "shared": "pair_aware_fixed_plus_shared_log",
            "by_order": "pair_aware_fixed_plus_order_log",
        }[slope_mode]
        return cls(n_edps, slope_mode, coefficients, ridge_penalty, label)

    def predict_capture(self, subset_masks: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        orders = np.array([int(mask).bit_count() for mask in subset_masks], dtype=np.int8)
        design = _pair_design(subset_masks, orders, log_scale, self.n_edps, self.slope_mode)
        return np.clip(_sigmoid(design @ self.coefficients), 1e-5, 1.0 - 1e-5)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": "pair_aware_fixed_plus_log",
            "n_edps": self.n_edps,
            "slope_mode": self.slope_mode,
            "parameter_count": self.parameter_count,
            "ridge_penalty": self.ridge_penalty,
            "coefficients": self.coefficients.tolist(),
        }


def _pair_design(
    subset_masks: np.ndarray,
    subset_orders: np.ndarray,
    log_scale: np.ndarray,
    n_edps: int,
    slope_mode: str,
) -> np.ndarray:
    pairs = tuple(itertools.combinations(range(n_edps), 2))
    order_columns = n_edps - 1
    pair_columns = len(pairs) - 1
    slope_columns = 0 if slope_mode == "none" else 1 if slope_mode == "shared" else n_edps - 1
    design = np.zeros((len(subset_masks), order_columns + pair_columns + slope_columns), dtype=float)

    for row, (mask, order, scale) in enumerate(zip(subset_masks, subset_orders, log_scale)):
        design[row, int(order) - 2] = 1.0
        included = np.array(
            [bool(mask & (1 << i) and mask & (1 << j)) for i, j in pairs],
            dtype=float,
        )
        included /= max(float(included.sum()), 1.0)
        # The last pair coefficient is minus the sum of the free coefficients,
        # which centers all pair affinities and makes the model identifiable.
        design[row, order_columns : order_columns + pair_columns] = included[:-1] - included[-1]
        if slope_mode == "shared":
            design[row, -1] = scale
        elif slope_mode == "by_order":
            design[row, order_columns + pair_columns + int(order) - 2] = scale
    return design


@dataclass(frozen=True)
class LatentMixtureModel(CalibrationModel):
    n_edps: int
    class_weight: float
    low_link: np.ndarray
    high_link: np.ndarray
    objective_cost: float
    name: str = "two_group_latent_mixture"

    @property
    def parameter_count(self) -> int:
        return 1 + 2 * self.n_edps

    @classmethod
    def fit(
        cls,
        dataset: CalibrationDataset,
        n_edps: int,
        seed: int,
    ) -> "LatentMixtureModel":
        masks, target, weight = _aggregate_by_subset(dataset)
        membership = np.array(
            [[bool(mask & (1 << i)) for i in range(n_edps)] for mask in masks],
            dtype=float,
        )
        normalized_weight = np.sqrt(weight / max(float(np.median(weight)), 1.0))
        normalized_weight = np.clip(normalized_weight, 0.25, 12.0)
        normalized_weight /= np.sqrt(
            np.array([comb(n_edps, int(mask).bit_count()) for mask in masks], dtype=float)
        )

        def unpack(theta: np.ndarray):
            pi = float(_sigmoid(theta[:1])[0])
            low = _sigmoid(theta[1 : 1 + n_edps])
            gap = _sigmoid(theta[1 + n_edps :])
            high = low + (1.0 - low) * gap
            return pi, low, high

        def prediction(theta: np.ndarray):
            pi, low, high = unpack(theta)
            low_product = np.prod(np.where(membership > 0, low[None, :], 1.0), axis=1)
            high_product = np.prod(np.where(membership > 0, high[None, :], 1.0), axis=1)
            return np.clip(pi * low_product + (1.0 - pi) * high_product, 1e-8, 1.0)

        def residual(theta: np.ndarray):
            predicted = prediction(theta)
            data_residual = normalized_weight * (np.log(predicted) - np.log(np.clip(target, 1e-8, 1.0)))
            regularization = 0.015 * theta
            return np.concatenate([data_residual, regularization])

        rng = np.random.default_rng(seed)
        best = None
        for restart in range(5):
            base = -1.0 + 0.25 * rng.normal(size=n_edps)
            theta = np.concatenate(
                [
                    np.array([rng.normal(scale=0.35)]),
                    base - rng.uniform(0.4, 1.0),
                    rng.normal(loc=0.2, scale=0.4, size=n_edps),
                ]
            )
            result = least_squares(residual, theta, max_nfev=2000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if best is None or result.cost < best.cost:
                best = result
        if best is None:
            raise RuntimeError("mixture fit did not run")
        pi, low, high = unpack(best.x)
        return cls(n_edps, pi, low, high, float(best.cost))

    def predict_capture(self, subset_masks: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        del log_scale
        result = np.ones(len(subset_masks), dtype=float)
        for row, mask in enumerate(subset_masks):
            selected = [i for i in range(self.n_edps) if int(mask) & (1 << i)]
            low_product = float(np.prod(self.low_link[selected]))
            high_product = float(np.prod(self.high_link[selected]))
            result[row] = self.class_weight * low_product + (1.0 - self.class_weight) * high_product
        return np.clip(result, 1e-8, 1.0)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": "latent_matchability_mixture",
            "n_edps": self.n_edps,
            "parameter_count": self.parameter_count,
            "class_weight": self.class_weight,
            "low_link": self.low_link.tolist(),
            "high_link": self.high_link.tolist(),
            "objective_cost": self.objective_cost,
        }


def _aggregate_by_subset(dataset: CalibrationDataset):
    values: dict[int, list[tuple[float, float]]] = {}
    balanced_weight = _balance_campaign_weights(dataset.campaign_ids, dataset.weight)
    for mask, signal, k0, weight in zip(
        dataset.subset_masks,
        dataset.signal,
        dataset.k0,
        balanced_weight,
    ):
        values.setdefault(int(mask), []).append((float(signal / max(k0, 1.0)), float(weight)))
    masks = np.array(sorted(values), dtype=np.int64)
    target = np.zeros(len(masks), dtype=float)
    weight = np.zeros(len(masks), dtype=float)
    for index, mask in enumerate(masks):
        rows = values[int(mask)]
        row_values = np.clip(np.array([item[0] for item in rows]), 1e-8, 1.0)
        row_weights = np.array([item[1] for item in rows])
        target[index] = float(np.average(row_values, weights=row_weights))
        weight[index] = float(row_weights.sum())
    return masks, target, weight
