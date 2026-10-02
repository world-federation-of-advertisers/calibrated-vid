"""Deterministic parameter sweeps for the two behavioral-market notebooks."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from calibrated_vid.behavioral_market import (
    MarketConfig,
    behavioral_diagnostics,
    simulate_market as simulate_stress_market,
)
from calibrated_vid.intuitive_market import (
    IntuitiveConfig,
    diagnostics as intuitive_diagnostics,
    simulate_market as simulate_intuitive_market,
)


def _metrics(report: dict[str, dict[str, float | int]]) -> dict[str, float]:
    return {
        "objective_gap_points": -float(
            report["paired_objective_test"]["traffic_minus_reach_median_points"]
        ),
        "direction_gap_points": float(
            report["paired_direction_test"][
                "medium_large_minus_large_medium_median_points"
            ]
        ),
        "large_reach_width_points": float(
            report["within_large_reach"]["overlap_p10_p90_width_points"]
        ),
    }


def _intuitive_metrics(report: dict[str, object]) -> dict[str, float]:
    return {
        "objective_gap_points": -float(
            report["objective"]["traffic_minus_reach_median_points"]  # type: ignore[index]
        ),
        "direction_gap_points": float(
            report["direction"][  # type: ignore[index]
                "medium_large_minus_large_medium_median_points"
            ]
        ),
        "large_reach_width_points": float(
            report["large_profiles"]["p10_p90_width_points"]  # type: ignore[index]
        ),
    }


def _stress_base(seed: int) -> MarketConfig:
    return MarketConfig(
        population_size=45_000,
        medium_reach=2_250,
        large_reach=9_000,
        objective_comparisons=12,
        traffic_comparisons=12,
        conversion_comparisons=0,
        direction_comparisons=12,
        random_seed=seed,
    )


def _intuitive_base(seed: int) -> IntuitiveConfig:
    return IntuitiveConfig(
        population_size=45_000,
        medium_reach=2_700,
        large_reach=9_000,
        training_replicates=6,
        objective_replicates_per_profile=3,
        direction_replicates=12,
        large_profile_replicates=4,
        random_seed=seed,
    )


def _row(model: str, sweep: str, parameter: str, value: float, seed: int, metrics: dict[str, float]) -> dict[str, float | int | str]:
    return {
        "model": model,
        "sweep": sweep,
        "parameter": parameter,
        "value": value,
        "seed": seed,
        **metrics,
    }


def run_one_factor_sweeps() -> list[dict[str, float | int | str]]:
    """Varies one interpretable lever at a time around each notebook's default."""

    rows: list[dict[str, float | int | str]] = []
    seeds = (20261002, 20261003, 20261004)
    stress_values = {
        "traffic_response_exponent": (0.5, 1.0, 1.5, 2.0, 2.5),
        "direction_cost_contrast": (0.0, 0.35, 0.70, 1.05, 1.40),
        "opportunity_shift_sigma": (0.0, 0.20, 0.40, 0.70, 1.00),
        "price_shift_sigma": (0.0, 0.20, 0.40, 0.60, 0.80),
        "selection_concentration": (0.75, 1.25, 2.0, 3.0, 4.0),
    }
    intuitive_values = {
        "response_contrast": (0.0, 0.25, 0.50, 0.75, 1.00, 1.25),
        "activity_contrast": (0.0, 0.25, 0.50, 0.75, 1.00, 1.25),
        "profile_strength": (0.0, 0.25, 0.50, 0.75, 1.00, 1.25),
        "selection_concentration": (4.0, 6.0, 8.0, 10.0, 12.0, 14.0),
        "traffic_response_exponent": (0.30, 0.40, 0.50, 0.60, 0.80, 1.00),
    }
    for seed in seeds:
        stress_base = _stress_base(seed)
        for parameter, values in stress_values.items():
            for value in values:
                config = replace(stress_base, **{parameter: value})
                metrics = _metrics(behavioral_diagnostics(simulate_stress_market(config)))
                rows.append(_row("stress", "one_factor", parameter, value, seed, metrics))

        intuitive_base = _intuitive_base(seed)
        for parameter, values in intuitive_values.items():
            for value in values:
                config = replace(intuitive_base, **{parameter: value})
                metrics = _intuitive_metrics(
                    intuitive_diagnostics(simulate_intuitive_market(config))
                )
                rows.append(
                    _row("intuitive", "one_factor", parameter, value, seed, metrics)
                )
    return rows


def run_joint_sweeps(
    samples: int = 48,
    regime: str = "wide",
) -> list[dict[str, float | int | str]]:
    """Samples combinations to reveal interactions hidden by one-factor curves.

    The wide regime includes null and weak mechanisms.  The active regime asks
    whether the qualitative results persist once every proposed mechanism is
    present at a moderate level; both are reported to avoid cherry-picking.
    """

    if regime not in {"wide", "active"}:
        raise ValueError("regime must be 'wide' or 'active'")

    rng = np.random.default_rng(20261002)
    rows: list[dict[str, float | int | str]] = []
    for ordinal in range(samples):
        seed = 20262000 + ordinal
        if regime == "wide":
            stress_ranges = {
                "traffic_response_exponent": (0.5, 2.5),
                "direction_cost_contrast": (0.0, 1.6),
                "opportunity_shift_sigma": (0.0, 1.0),
                "price_shift_sigma": (0.0, 0.9),
                "selection_concentration": (0.75, 4.0),
            }
            intuitive_ranges = {
                "activity_contrast": (0.0, 1.25),
                "response_contrast": (0.0, 1.25),
                "profile_strength": (0.0, 1.25),
                "traffic_response_exponent": (0.3, 1.0),
                "selection_concentration": (4.0, 14.0),
            }
        else:
            stress_ranges = {
                "traffic_response_exponent": (0.8, 2.2),
                "direction_cost_contrast": (0.5, 1.5),
                "opportunity_shift_sigma": (0.25, 0.85),
                "price_shift_sigma": (0.2, 0.75),
                "selection_concentration": (1.25, 3.0),
            }
            intuitive_ranges = {
                "activity_contrast": (0.6, 1.2),
                "response_contrast": (0.6, 1.2),
                "profile_strength": (0.5, 1.2),
                "traffic_response_exponent": (0.35, 0.7),
                "selection_concentration": (8.0, 14.0),
            }
        stress = replace(
            _stress_base(seed),
            **{
                key: float(rng.uniform(*bounds))
                for key, bounds in stress_ranges.items()
            },
        )
        stress_metrics = _metrics(
            behavioral_diagnostics(simulate_stress_market(stress))
        )
        rows.append(
            {
                "model": "stress",
                "sweep": "joint",
                "regime": regime,
                "sample": ordinal + 1,
                "seed": seed,
                **{
                    key: value
                    for key, value in asdict(stress).items()
                    if key
                    in {
                        "traffic_response_exponent",
                        "direction_cost_contrast",
                        "opportunity_shift_sigma",
                        "price_shift_sigma",
                        "selection_concentration",
                    }
                },
                **stress_metrics,
            }
        )

        intuitive = replace(
            _intuitive_base(seed),
            **{
                key: float(rng.uniform(*bounds))
                for key, bounds in intuitive_ranges.items()
            },
        )
        intuitive_metrics = _intuitive_metrics(
            intuitive_diagnostics(simulate_intuitive_market(intuitive))
        )
        rows.append(
            {
                "model": "intuitive",
                "sweep": "joint",
                "regime": regime,
                "sample": ordinal + 1,
                "seed": seed,
                **{
                    key: value
                    for key, value in asdict(intuitive).items()
                    if key
                    in {
                        "activity_contrast",
                        "response_contrast",
                        "profile_strength",
                        "traffic_response_exponent",
                        "selection_concentration",
                    }
                },
                **intuitive_metrics,
            }
        )
    return rows


def summarize_joint(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    summaries = []
    for model in ("stress", "intuitive"):
        for regime in ("wide", "active"):
            cohort = [
                row
                for row in rows
                if row["model"] == model
                and row["sweep"] == "joint"
                and row["regime"] == regime
            ]
            if not cohort:
                continue
            objective = np.asarray([float(row["objective_gap_points"]) for row in cohort])
            direction = np.asarray([float(row["direction_gap_points"]) for row in cohort])
            width = np.asarray([float(row["large_reach_width_points"]) for row in cohort])
            summaries.append(
                {
                    "model": model,
                    "regime": regime,
                    "configurations": len(cohort),
                    "traffic_lower_share": float(np.mean(objective > 0.0)),
                    "positive_direction_share": float(np.mean(direction > 0.0)),
                    "large_width_over_5_share": float(np.mean(width > 5.0)),
                    "all_qualitative_share": float(
                        np.mean((objective > 0.0) & (direction > 0.0) & (width > 5.0))
                    ),
                    "all_material_share": float(
                        np.mean((objective > 5.0) & (direction > 5.0) & (width > 10.0))
                    ),
                    "objective_gap_median": float(np.median(objective)),
                    "direction_gap_median": float(np.median(direction)),
                    "large_width_median": float(np.median(width)),
                }
            )
    return summaries


def run_sensitivity(output_directory: str | Path) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    one_factor = run_one_factor_sweeps()
    joint = run_joint_sweeps(regime="wide") + run_joint_sweeps(regime="active")
    summary = summarize_joint(joint)
    _write_csv(output / "one_factor_sweep.csv", one_factor)
    _write_csv(output / "joint_sweep.csv", joint)
    _write_csv(output / "joint_summary.csv", summary)
    report: dict[str, object] = {"joint_summary": summary}
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
