from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .calibrated_venn_labeling import (
    _calibration_specs,
    _training_campaigns,
    label_from_cumulative_targets,
    reconcile_reachable_cells,
    reconcile_reachable_cells_greedy,
)
from .config import SimulationConfig
from .daily_labeling import _report_union, _truth_union, generate_temporal_stress_campaign, report_specs
from .joint_decoding import calibrate_report_pairwise_maximum_entropy
from .measurement import measure_report
from .panel_validation import _fit_reference_models, _panel_observations, draw_panel
from .population import generate_campaign, make_world


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def run_temporal_solver_benchmark(output_dir: Path) -> dict:
    config = SimulationConfig(
        n_users=4_000,
        population_size=180_000_000,
        n_edps=10,
        n_weeks=13,
        panel_size=4_000,
        minimum_calibration_intersection=30_000.0,
        seed=20260901,
    )
    world = make_world(config)
    training = _training_campaigns(world, copies=1)
    specs = _calibration_specs(config.n_weeks, config.n_edps)
    panel = draw_panel(world, "representative", config.seed + 100, panel_size=4_000)
    models = _fit_reference_models(
        config,
        list(_panel_observations(world, training, specs, panel).values()),
        seed_offset=991,
    )
    campaigns = [
        generate_campaign(world, "website_retargeting", config.seed + 201, "solver_website"),
        generate_campaign(world, "crm_customer_list", config.seed + 202, "solver_crm"),
        generate_temporal_stress_campaign(
            world,
            "shared_seed_then_expansion",
            config.seed + 203,
            "solver_shared_seed",
        ),
    ]
    all_edps = tuple(range(config.n_edps))
    rows: list[dict] = []
    solver_functions = {
        "exact_milp": reconcile_reachable_cells,
        "fast_greedy": reconcile_reachable_cells_greedy,
    }

    for campaign in campaigns:
        observations = [
            measure_report(world, campaign, tuple(range(day + 1)), all_edps)
            for day in range(config.n_weeks)
        ]
        decoded = [
            calibrate_report_pairwise_maximum_entropy(
                observation,
                models["fixed_log"],
                pair_ridge=1e-6,
                evidence_half_saturation=5.0,
                name="solver_benchmark",
            )
            for observation in observations
        ]
        for solver_name, solver in solver_functions.items():
            current = np.zeros(1 << config.n_edps, dtype=int)
            current[0] = config.n_users
            targets: list[np.ndarray] = []
            for day, (observation, report) in enumerate(zip(observations, decoded)):
                marginals = np.asarray(
                    [
                        int(
                            round(
                                observation.baseline_intersections[1 << edp]
                                / observation.person_weight
                            )
                        )
                        for edp in range(config.n_edps)
                    ],
                    dtype=int,
                )
                reconciled = solver(
                    current,
                    report.exclusive_cells / observation.person_weight,
                    marginals,
                )
                targets.append(reconciled.target_cells)
                rows.extend(
                    [
                        {
                            "campaign": campaign.campaign_id,
                            "solver": solver_name,
                            "metric": "solve_seconds",
                            "day": day + 1,
                            "value": reconciled.solve_seconds,
                        },
                        {
                            "campaign": campaign.campaign_id,
                            "solver": solver_name,
                            "metric": "union_adjustment",
                            "day": day + 1,
                            "value": reconciled.union_adjustment,
                        },
                        {
                            "campaign": campaign.campaign_id,
                            "solver": solver_name,
                            "metric": "cell_l1_adjustment",
                            "day": day + 1,
                            "value": reconciled.cell_l1_adjustment,
                        },
                    ]
                )
                current = reconciled.target_cells
            labeled = label_from_cumulative_targets(
                campaign,
                targets,
                f"solver_{solver_name}",
                timing_policy="active_today",
            )
            for report_name, weeks, edps in report_specs(config.n_edps, config.n_weeks):
                truth = _truth_union(campaign, weeks, edps)
                estimate = _report_union(labeled.labels, weeks, edps)
                rows.append(
                    {
                        "campaign": campaign.campaign_id,
                        "solver": solver_name,
                        "metric": "report_error",
                        "day": max(weeks) + 1,
                        "value": abs(estimate - truth) / max(truth, 1),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "temporal_solver_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        solver: {
            metric: _summary(
                [
                    row["value"]
                    for row in rows
                    if row["solver"] == solver and row["metric"] == metric
                ]
            )
            for metric in sorted({row["metric"] for row in rows if row["solver"] == solver})
        }
        for solver in solver_functions
    }
    (output_dir / "temporal_solver_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run_temporal_solver_benchmark(arguments.output_dir)
