from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .calibrated_venn_labeling import (
    _calibration_specs,
    _evaluation_campaigns,
    _training_campaigns,
)
from .config import SimulationConfig
from .joint_decoding import calibrate_report_pairwise_maximum_entropy
from .measurement import measure_report
from .panel_validation import _fit_reference_models, _panel_observations, draw_panel
from .population import make_world


HALF_SATURATION_VALUES = (5.0, 20.0, 50.0, 100.0, 250.0, 500.0, 1_000.0)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def run_online_regularization_sweep(output_dir: Path) -> dict[str, object]:
    config = SimulationConfig(
        n_users=12_000,
        population_size=180_000_000,
        n_edps=10,
        n_weeks=13,
        panel_size=5_000,
        minimum_calibration_intersection=30_000.0,
        ridge_penalty=0.35,
        seed=20260831,
    )
    world = make_world(config)
    training = _training_campaigns(world, copies=2)
    campaigns = _evaluation_campaigns(world)
    panel = draw_panel(world, "representative", config.seed + 400_000, panel_size=5_000)
    model = _fit_reference_models(
        config,
        list(
            _panel_observations(
                world,
                training,
                _calibration_specs(config.n_weeks, config.n_edps),
                panel,
            ).values()
        ),
        seed_offset=811,
    )["fixed_log"]
    all_edps = tuple(range(config.n_edps))
    rows: list[dict[str, object]] = []

    for campaign in campaigns:
        observations = [
            measure_report(world, campaign, tuple(range(day + 1)), all_edps)
            for day in range(config.n_weeks)
        ]
        baseline_previous = 0.0
        for day, observation in enumerate(observations):
            truth = float(observation.truth_unions[-1])
            baseline = float(observation.baseline_unions[-1])
            baseline_online = max(baseline_previous, baseline)
            baseline_previous = baseline_online
            rows.append(
                {
                    "campaign": campaign.campaign_id,
                    "scenario": campaign.scenario,
                    "half_saturation": "existing_vid",
                    "day": day + 1,
                    "raw_union": baseline,
                    "online_union": baseline_online,
                    "truth_union": truth,
                    "raw_error": abs(baseline - truth) / max(truth, 1.0),
                    "online_error": abs(baseline_online - truth) / max(truth, 1.0),
                    "downward_step": 0.0,
                }
            )

        for half_saturation in HALF_SATURATION_VALUES:
            previous = 0.0
            prior_raw = 0.0
            for day, observation in enumerate(observations):
                decoded = calibrate_report_pairwise_maximum_entropy(
                    observation,
                    model,
                    pair_ridge=1e-6,
                    evidence_half_saturation=half_saturation,
                    name=f"half_{half_saturation:g}",
                )
                raw = float(decoded.full_union)
                online = max(previous, raw)
                truth = float(observation.truth_unions[-1])
                rows.append(
                    {
                        "campaign": campaign.campaign_id,
                        "scenario": campaign.scenario,
                        "half_saturation": half_saturation,
                        "day": day + 1,
                        "raw_union": raw,
                        "online_union": online,
                        "truth_union": truth,
                        "raw_error": abs(raw - truth) / max(truth, 1.0),
                        "online_error": abs(online - truth) / max(truth, 1.0),
                        "downward_step": max(prior_raw - raw, 0.0) / max(prior_raw, 1.0),
                    }
                )
                previous = online
                prior_raw = raw

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "online_regularization_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {"configuration": config.__dict__, "settings": {}}
    for setting in ("existing_vid", *HALF_SATURATION_VALUES):
        selected = [row for row in rows if row["half_saturation"] == setting]
        downward = [float(row["downward_step"]) for row in selected]
        summary["settings"][str(setting)] = {
            "raw_error": _summary([float(row["raw_error"]) for row in selected]),
            "online_monotone_error": _summary(
                [float(row["online_error"]) for row in selected]
            ),
            "downward_step_fraction": float(np.mean(np.asarray(downward) > 1e-9)),
            "downward_step": _summary(downward),
        }
    (output_dir / "online_regularization_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run_online_regularization_sweep(arguments.output_dir)
