from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    n_users: int = 12_000
    population_size: int = 120_000_000
    n_edps: int = 10
    n_weeks: int = 13
    reference_pool_size: int = 5_000_000_000
    calibration_train_campaigns: int = 10
    calibration_holdout_campaigns: int = 4
    stress_campaigns_per_scenario: int = 2
    request_order_trials: int = 4
    seed: int = 20260825
    minimum_calibration_intersection: float = 25_000.0
    ridge_penalty: float = 0.35
    material_holdout_improvement: float = 0.005
    review_movement_fraction: float = 0.05
    panel_size: int = 5_000
    panel_draws: int = 4
    panel_activation_improvement: float = 0.005

    @property
    def person_weight(self) -> float:
        return self.population_size / self.n_users

    @classmethod
    def for_profile(cls, profile: str) -> "SimulationConfig":
        if profile == "quick":
            return cls()
        if profile == "full":
            return cls(
                n_users=30_000,
                calibration_train_campaigns=24,
                calibration_holdout_campaigns=8,
                stress_campaigns_per_scenario=6,
                request_order_trials=12,
                minimum_calibration_intersection=50_000.0,
                panel_draws=12,
            )
        raise ValueError(f"unknown profile: {profile}")
