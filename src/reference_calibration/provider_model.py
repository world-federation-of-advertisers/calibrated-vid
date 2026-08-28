from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .measurement import ReportObservation
from .population import AUDIENCE_STRATEGIES, CAMPAIGN_OBJECTIVES
from .sets import members, project_to_bounded_sum


def _report_features(
    observation: ReportObservation,
    n_edps: int,
    include_context: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build observable aggregate features for the provider's total model."""
    population = float(observation.truth_intersections[0])
    local_n = len(observation.edps)
    marginals = np.asarray(
        [observation.truth_intersections[1 << local] for local in range(local_n)],
        dtype=float,
    )
    fractions = marginals / population
    baseline = float(observation.baseline_unions[-1]) / population
    lower = float(np.max(marginals)) / population
    upper = float(min(np.sum(marginals), population)) / population

    values = [
        local_n / n_edps,
        float(np.sum(fractions)),
        float(np.mean(fractions)),
        float(np.std(fractions)),
        float(np.min(fractions)),
        float(np.max(fractions)),
        baseline,
        lower,
        upper,
        float(np.log(max(np.exp(np.mean(np.log(np.maximum(fractions, 1e-9)))), 1e-9))),
    ]
    names = [
        "edp_count_fraction",
        "sum_reach_fraction",
        "mean_reach_fraction",
        "std_reach_fraction",
        "min_reach_fraction",
        "max_reach_fraction",
        "baseline_union_fraction",
        "lower_union_fraction",
        "upper_union_fraction",
        "log_geometric_mean_reach",
    ]

    global_to_local = {global_edp: local for local, global_edp in enumerate(observation.edps)}
    for edp in range(n_edps):
        present = edp in global_to_local
        values.append(1.0 if present else 0.0)
        values.append(float(fractions[global_to_local[edp]]) if present else 0.0)
        names.extend((f"edp_{edp}_present", f"edp_{edp}_reach"))

    pair_email_to_min: list[float] = []
    pair_email_to_baseline: list[float] = []
    for left, right in combinations(range(n_edps), 2):
        if left not in global_to_local or right not in global_to_local:
            values.extend((0.0, 0.0))
            names.extend(
                (
                    f"pair_{left}_{right}_email_to_min",
                    f"pair_{left}_{right}_email_to_baseline",
                )
            )
            continue
        local_mask = (1 << global_to_local[left]) | (1 << global_to_local[right])
        email_overlap = max(float(observation.email_intersections[local_mask]), 0.0)
        minimum = max(
            min(
                float(observation.truth_intersections[1 << global_to_local[left]]),
                float(observation.truth_intersections[1 << global_to_local[right]]),
            ),
            observation.person_weight,
        )
        baseline_pair = max(
            float(observation.baseline_intersections[local_mask]),
            observation.person_weight,
        )
        email_to_min = email_overlap / minimum
        email_to_baseline = np.log1p(email_overlap / baseline_pair)
        pair_email_to_min.append(email_to_min)
        pair_email_to_baseline.append(email_to_baseline)
        values.extend((email_to_min, email_to_baseline))
        names.extend(
            (
                f"pair_{left}_{right}_email_to_min",
                f"pair_{left}_{right}_email_to_baseline",
            )
        )

    for label, observed in (
        ("pair_email_to_min", pair_email_to_min),
        ("pair_email_to_baseline", pair_email_to_baseline),
    ):
        array = np.asarray(observed, dtype=float)
        if len(array):
            values.extend(
                (
                    float(np.mean(array)),
                    float(np.std(array)),
                    float(np.min(array)),
                    float(np.max(array)),
                )
            )
        else:
            values.extend((0.0, 0.0, 0.0, 0.0))
        names.extend(
            (
                f"{label}_mean",
                f"{label}_std",
                f"{label}_min",
                f"{label}_max",
            )
        )

    for order_label, selector in (
        ("two_way", lambda order: order == 2),
        ("three_way", lambda order: order == 3),
        ("four_plus", lambda order: order >= 4),
    ):
        ratios: list[float] = []
        email_fraction = 0.0
        for local_mask in range(1, len(observation.global_masks)):
            order = local_mask.bit_count()
            if not selector(order):
                continue
            email_overlap = max(float(observation.email_intersections[local_mask]), 0.0)
            baseline_intersection = max(
                float(observation.baseline_intersections[local_mask]),
                observation.person_weight,
            )
            ratios.append(np.log1p(email_overlap / baseline_intersection))
            email_fraction += email_overlap / population
        values.extend(
            (
                email_fraction,
                float(np.mean(ratios)) if ratios else 0.0,
                float(np.std(ratios)) if ratios else 0.0,
            )
        )
        names.extend(
            (
                f"{order_label}_email_fraction",
                f"{order_label}_ratio_mean",
                f"{order_label}_ratio_std",
            )
        )

    if include_context:
        reach_weights = fractions / max(float(fractions.sum()), 1e-9)
        for objective in CAMPAIGN_OBJECTIVES:
            values.append(float(np.mean([item == objective for item in observation.objectives])))
            values.append(
                float(
                    sum(
                        reach_weights[index]
                        for index, item in enumerate(observation.objectives)
                        if item == objective
                    )
                )
            )
            names.extend((f"objective_{objective}_share", f"objective_{objective}_reach_share"))
        for strategy in AUDIENCE_STRATEGIES:
            values.append(
                float(np.mean([item == strategy for item in observation.audience_strategies]))
            )
            values.append(
                float(
                    sum(
                        reach_weights[index]
                        for index, item in enumerate(observation.audience_strategies)
                        if item == strategy
                    )
                )
            )
            names.extend((f"strategy_{strategy}_share", f"strategy_{strategy}_reach_share"))

    return np.asarray(values, dtype=float), tuple(names)


@dataclass(frozen=True)
class PanelTotalReachModel:
    """Aggregate surrogate for an email-first demographic-agnostic VID model.

    The model uses email-derived VID overlap, per-EDP reach, and optional
    campaign context.  It never uses the calibration-only Reference-ID counts.
    """

    name: str
    n_edps: int
    include_context: bool
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    training_features: np.ndarray
    target_mean: float
    kernel_weights: np.ndarray
    bandwidth: float
    ridge_penalty: float

    @classmethod
    def fit(
        cls,
        observations: list[ReportObservation],
        n_edps: int,
        include_context: bool,
        ridge_penalty: float = 0.08,
    ) -> "PanelTotalReachModel":
        feature_rows: list[np.ndarray] = []
        targets: list[float] = []
        feature_names: tuple[str, ...] | None = None
        for observation in observations:
            features, names = _report_features(observation, n_edps, include_context)
            feature_names = names
            truth = max(float(observation.truth_unions[-1]), 1.0)
            baseline = max(float(observation.baseline_unions[-1]), 1.0)
            feature_rows.append(features)
            targets.append(float(np.clip(np.log(truth / baseline), -3.0, 3.0)))
        matrix = np.vstack(feature_rows)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (matrix - mean) / scale
        distances = np.sum(
            (standardized[:, None, :] - standardized[None, :, :]) ** 2,
            axis=2,
        )
        nonzero = np.sqrt(distances[distances > 1e-12])
        bandwidth = float(np.median(nonzero)) if len(nonzero) else 1.0
        bandwidth = max(bandwidth, 1.0)
        kernel = np.exp(-distances / (2.0 * bandwidth * bandwidth))
        target_array = np.asarray(targets, dtype=float)
        target_mean = float(target_array.mean())
        weights = np.linalg.solve(
            kernel + ridge_penalty * np.eye(len(kernel)),
            target_array - target_mean,
        )
        return cls(
            name=(
                "provider_panel_total_with_context"
                if include_context
                else "provider_panel_total_email_only"
            ),
            n_edps=n_edps,
            include_context=include_context,
            feature_names=feature_names or (),
            feature_mean=mean,
            feature_scale=scale,
            training_features=standardized,
            target_mean=target_mean,
            kernel_weights=weights,
            bandwidth=bandwidth,
            ridge_penalty=ridge_penalty,
        )

    @property
    def parameter_count(self) -> int:
        return int(
            self.training_features.size
            + len(self.kernel_weights)
            + 2 * len(self.feature_names)
            + 2
        )

    def predict(self, observation: ReportObservation) -> float:
        features, names = _report_features(observation, self.n_edps, self.include_context)
        if names != self.feature_names:
            raise ValueError("provider-model feature schema mismatch")
        standardized = (features - self.feature_mean) / self.feature_scale
        distances = np.sum((self.training_features - standardized[None, :]) ** 2, axis=1)
        kernel = np.exp(-distances / (2.0 * self.bandwidth * self.bandwidth))
        correction = self.target_mean + float(kernel @ self.kernel_weights)
        baseline = float(observation.baseline_unions[-1])
        estimate = baseline * float(np.exp(np.clip(correction, -3.0, 3.0)))
        marginals = np.asarray(
            [
                observation.truth_intersections[1 << local]
                for local in range(len(observation.edps))
            ],
            dtype=float,
        )
        lower = float(np.max(marginals))
        upper = float(min(np.sum(marginals), observation.truth_intersections[0]))
        return float(np.clip(estimate, lower, upper))


def _allocator_features(
    observation: ReportObservation,
    n_edps: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    population = float(observation.truth_intersections[0])
    marginals = np.asarray(
        [
            observation.truth_intersections[1 << local]
            for local in range(len(observation.edps))
        ],
        dtype=float,
    )
    fractions = marginals / population
    values = [
        len(observation.edps) / n_edps,
        float(np.sum(fractions)),
        float(np.mean(fractions)),
        float(np.std(fractions)),
        float(np.min(fractions)),
        float(np.max(fractions)),
        float(observation.baseline_unions[-1]) / population,
    ]
    names = [
        "edp_count_fraction",
        "sum_reach_fraction",
        "mean_reach_fraction",
        "std_reach_fraction",
        "min_reach_fraction",
        "max_reach_fraction",
        "baseline_union_fraction",
    ]
    reach_weights = fractions / max(float(fractions.sum()), 1e-9)
    for objective in CAMPAIGN_OBJECTIVES:
        values.append(float(np.mean([item == objective for item in observation.objectives])))
        values.append(
            float(
                sum(
                    reach_weights[index]
                    for index, item in enumerate(observation.objectives)
                    if item == objective
                )
            )
        )
        names.extend((f"objective_{objective}_share", f"objective_{objective}_reach_share"))
    for strategy in AUDIENCE_STRATEGIES:
        values.append(
            float(np.mean([item == strategy for item in observation.audience_strategies]))
        )
        values.append(
            float(
                sum(
                    reach_weights[index]
                    for index, item in enumerate(observation.audience_strategies)
                    if item == strategy
                )
            )
        )
        names.extend((f"strategy_{strategy}_share", f"strategy_{strategy}_reach_share"))
    baseline = np.maximum(observation.baseline_demographic_union, 1e-9)
    shares = baseline / float(baseline.sum())
    return (
        np.concatenate([np.asarray(values, dtype=float), shares]),
        tuple(names) + tuple(f"baseline_demo_share_{index}" for index in range(len(shares))),
    )


class DemographicAllocator:
    name: str

    def target_shares(self, observation: ReportObservation) -> np.ndarray:
        raise NotImplementedError

    def allocate(self, total_reach: float, observation: ReportObservation) -> np.ndarray:
        shares = self.target_shares(observation)
        return project_to_bounded_sum(
            total_reach * shares,
            total_reach,
            upper=observation.demographic_population,
        )


@dataclass(frozen=True)
class ProportionalDemographicAllocator(DemographicAllocator):
    name: str = "proportional_demographic_scaling"

    def target_shares(self, observation: ReportObservation) -> np.ndarray:
        baseline = np.maximum(observation.baseline_demographic_union, 1e-9)
        return baseline / float(baseline.sum())


@dataclass(frozen=True)
class FixedDemographicAllocator(DemographicAllocator):
    correction: np.ndarray
    name: str = "fixed_panel_demographic_adjustment"

    @classmethod
    def fit(
        cls,
        observations: list[ReportObservation],
        shrinkage: float = 0.25,
    ) -> "FixedDemographicAllocator":
        residuals = []
        for observation in observations:
            truth = np.maximum(observation.truth_demographic_union, 1e-9)
            truth /= float(truth.sum())
            baseline = np.maximum(observation.baseline_demographic_union, 1e-9)
            baseline /= float(baseline.sum())
            residuals.append(truth - baseline)
        correction = np.mean(np.vstack(residuals), axis=0) / (1.0 + shrinkage)
        return cls(correction=correction)

    def target_shares(self, observation: ReportObservation) -> np.ndarray:
        baseline = np.maximum(observation.baseline_demographic_union, 1e-9)
        baseline /= float(baseline.sum())
        return project_to_bounded_sum(
            baseline + self.correction,
            1.0,
            upper=np.ones_like(baseline),
        )


@dataclass(frozen=True)
class ContextualDemographicAllocator(DemographicAllocator):
    n_edps: int
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    ridge_penalty: float
    name: str = "contextual_panel_demographic_adjustment"

    @classmethod
    def fit(
        cls,
        observations: list[ReportObservation],
        n_edps: int,
        ridge_penalty: float = 40.0,
    ) -> "ContextualDemographicAllocator":
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        feature_names: tuple[str, ...] | None = None
        for observation in observations:
            row, names = _allocator_features(observation, n_edps)
            feature_names = names
            truth = np.maximum(observation.truth_demographic_union, 1e-9)
            truth /= float(truth.sum())
            baseline = np.maximum(observation.baseline_demographic_union, 1e-9)
            baseline /= float(baseline.sum())
            features.append(row)
            targets.append(truth - baseline)
        matrix = np.vstack(features)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (matrix - mean) / scale
        design = np.column_stack([np.ones(len(standardized)), standardized])
        penalty = ridge_penalty * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ np.vstack(targets),
        )
        return cls(
            n_edps=n_edps,
            feature_names=feature_names or (),
            feature_mean=mean,
            feature_scale=scale,
            coefficients=coefficients,
            ridge_penalty=ridge_penalty,
        )

    @property
    def parameter_count(self) -> int:
        return int(self.coefficients.size)

    def target_shares(self, observation: ReportObservation) -> np.ndarray:
        features, names = _allocator_features(observation, self.n_edps)
        if names != self.feature_names:
            raise ValueError("demographic-allocator feature schema mismatch")
        standardized = (features - self.feature_mean) / self.feature_scale
        design = np.concatenate([[1.0], standardized])
        correction = np.clip(design @ self.coefficients, -0.08, 0.08)
        baseline = np.maximum(observation.baseline_demographic_union, 1e-9)
        baseline /= float(baseline.sum())
        return project_to_bounded_sum(
            baseline + correction,
            1.0,
            upper=np.ones_like(baseline),
        )


def demographic_distribution_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate_share = np.asarray(estimate, dtype=float) / max(float(np.sum(estimate)), 1.0)
    truth_share = np.asarray(truth, dtype=float) / max(float(np.sum(truth)), 1.0)
    return float(0.5 * np.sum(np.abs(estimate_share - truth_share)))


def demographic_reach_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sum(np.abs(np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)))
        / max(float(np.sum(truth)), 1.0)
    )
