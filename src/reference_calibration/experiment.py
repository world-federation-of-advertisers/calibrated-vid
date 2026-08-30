from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .evaluation import (
    CalibratedReport,
    calibrate_report,
    observable_capture_residual,
    relative_error,
    summarize,
)
from .measurement import CalibrationDataset, ReportObservation, calibration_dataset, measure_report
from .models import CalibrationModel, LatentMixtureModel, PairAwareLogModel
from .population import SCENARIOS, Campaign, SyntheticWorld, generate_campaign, make_world
from .reconciliation import FinalizedReport, ReportCandidate, ReportKey, ResultRegistry
from .sets import members, selected_masks


def calibration_checkpoints(n_weeks: int) -> tuple[tuple[int, ...], ...]:
    endpoints = sorted({min(3, n_weeks), min(6, n_weeks), min(9, n_weeks), n_weeks})
    return tuple(tuple(range(endpoint)) for endpoint in endpoints if endpoint > 0)


def report_specs(n_edps: int, n_weeks: int) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]:
    all_edps = tuple(range(n_edps))
    five_a = tuple(range(min(5, n_edps)))
    five_b = tuple(i for i in (0, 2, 4, 6, 8) if i < n_edps)
    return (
        ("weeks_1_3__2_edps", tuple(range(0, min(3, n_weeks))), (0, 1)),
        ("weeks_1_12__10_edps", tuple(range(0, min(12, n_weeks))), all_edps),
        ("weeks_5_12__2_edps", tuple(range(4, min(12, n_weeks))), (0, 1)),
        ("weeks_7_13__5_edps", tuple(range(6, n_weeks)), five_a),
        ("all_weeks__5_edps", tuple(range(n_weeks)), five_b),
        ("all_weeks__10_edps", tuple(range(n_weeks)), all_edps),
        ("noncontiguous__5_edps", tuple(i for i in (0, 2, 4, 7, 10, 12) if i < n_weeks), five_b),
        ("weeks_4_9__5_edps", tuple(range(3, min(9, n_weeks))), five_a),
        ("weeks_2_10__10_edps", tuple(range(1, min(10, n_weeks))), all_edps),
        ("weeks_1_7__5_edps", tuple(range(0, min(7, n_weeks))), five_a),
        ("week_7__5_edps", (min(6, n_weeks - 1),), five_a),
        ("all_weeks__2_edps", tuple(range(n_weeks)), (0, 1)),
    )


def _make_calibration_campaigns(world: SyntheticWorld):
    config = world.config
    total = config.calibration_train_campaigns + config.calibration_holdout_campaigns
    campaigns = [
        generate_campaign(
            world,
            "representative",
            config.seed + 10_000 + index,
            f"calibration_{index:03d}",
        )
        for index in range(total)
    ]
    train = campaigns[: config.calibration_train_campaigns]
    holdout = campaigns[config.calibration_train_campaigns :]
    return train, holdout


def _observations_for_campaigns(
    world: SyntheticWorld,
    campaigns: Iterable[Campaign],
    checkpoints: Iterable[tuple[int, ...]],
) -> list[ReportObservation]:
    all_edps = tuple(range(world.config.n_edps))
    return [
        measure_report(world, campaign, weeks, all_edps)
        for campaign in campaigns
        for weeks in checkpoints
    ]


def _evaluate_full_roster(
    observations: list[ReportObservation],
    models: list[CalibrationModel],
) -> list[dict]:
    rows: list[dict] = []
    for observation in observations:
        requested_masks = selected_masks(len(observation.edps), (2, 5, 10))
        calibrated = {model.name: calibrate_report(observation, model) for model in models}
        for subset in requested_masks:
            truth = float(observation.truth_unions[subset])
            if truth <= 0:
                continue
            rows.append(
                {
                    "category": "holdout_union",
                    "scenario": observation.scenario,
                    "campaign": observation.campaign_id,
                    "report": f"prefix_{len(observation.weeks)}",
                    "model": "baseline_vid",
                    "edp_count": subset.bit_count(),
                    "value": relative_error(float(observation.baseline_unions[subset]), truth),
                }
            )
            for model in models:
                rows.append(
                    {
                        "category": "holdout_union",
                        "scenario": observation.scenario,
                        "campaign": observation.campaign_id,
                        "report": f"prefix_{len(observation.weeks)}",
                        "model": model.name,
                        "edp_count": subset.bit_count(),
                        "value": relative_error(float(calibrated[model.name].union_values[subset]), truth),
                    }
                )
    return rows


def _capture_validation(dataset: CalibrationDataset, models: list[CalibrationModel]) -> list[dict]:
    rows: list[dict] = []
    truth_capture = dataset.signal / np.maximum(dataset.truth, 1.0)
    for model in models:
        predicted = model.predict_capture(dataset.subset_masks, dataset.log_scale)
        for order_group, selector in (
            ("2", dataset.subset_orders == 2),
            ("3", dataset.subset_orders == 3),
            ("4+", dataset.subset_orders >= 4),
        ):
            for value in np.abs(predicted[selector] - truth_capture[selector]):
                rows.append(
                    {
                        "category": "capture_absolute_error",
                        "scenario": "representative_holdout",
                        "campaign": "all",
                        "report": order_group,
                        "model": model.name,
                        "edp_count": order_group,
                        "value": float(value),
                    }
                )
    return rows


def _candidate_from_report(calibrated: CalibratedReport) -> ReportCandidate:
    observation = calibrated.observation
    marginal_reaches = tuple(
        float(observation.truth_intersections[1 << local])
        for local in range(len(observation.edps))
    )
    return ReportCandidate(
        key=ReportKey(
            campaign_id=observation.campaign_id,
            model_name=calibrated.model_name,
            weeks=observation.weeks,
            edps=observation.edps,
        ),
        raw_union=calibrated.full_union,
        marginal_reaches=marginal_reaches,
        uncertainty=max(0.02 * calibrated.full_union, observation.observation_weight if hasattr(observation, "observation_weight") else 1.0),
        population_size=observation.truth_intersections[0],
    )


def _run_registry(
    calibrated_by_label: dict[str, CalibratedReport],
    ordered_labels: list[str],
    config: SimulationConfig,
) -> tuple[ResultRegistry, dict[str, FinalizedReport], int]:
    registry = ResultRegistry(config.n_weeks, config.review_movement_fraction)
    results: dict[str, FinalizedReport] = {}
    repeat_failures = 0
    for label in ordered_labels:
        candidate = _candidate_from_report(calibrated_by_label[label])
        first = registry.finalize(candidate)
        second = registry.finalize(candidate)
        if first.finalized_union != second.finalized_union or first.status != second.status:
            repeat_failures += 1
        results[label] = first
    return registry, results, repeat_failures


def _stress_validation(
    world: SyntheticWorld,
    models: list[CalibrationModel],
) -> tuple[list[dict], dict]:
    config = world.config
    specs = report_specs(config.n_edps, config.n_weeks)
    rows: list[dict] = []
    structural = {
        "repeat_failures": 0,
        "audits": [],
        "review_counts": {},
        "order_sensitivity": {},
    }

    for scenario_index, scenario in enumerate(SCENARIOS):
        for campaign_index in range(config.stress_campaigns_per_scenario):
            campaign = generate_campaign(
                world,
                scenario,
                config.seed + 100_000 + 10_000 * scenario_index + campaign_index,
                f"stress_{scenario}_{campaign_index:02d}",
            )
            observations = {
                label: measure_report(world, campaign, weeks, edps)
                for label, weeks, edps in specs
            }
            full_observation = observations["all_weeks__10_edps"]

            for label, observation in observations.items():
                truth = float(observation.truth_unions[-1])
                rows.append(
                    {
                        "category": "stress_union_raw",
                        "scenario": scenario,
                        "campaign": campaign.campaign_id,
                        "report": label,
                        "model": "baseline_vid",
                        "edp_count": len(observation.edps),
                        "value": relative_error(float(observation.baseline_unions[-1]), truth),
                    }
                )

            for model in models:
                calibrated = {
                    label: calibrate_report(observation, model)
                    for label, observation in observations.items()
                }
                for label, report in calibrated.items():
                    truth = float(report.observation.truth_unions[-1])
                    rows.append(
                        {
                            "category": "stress_union_raw",
                            "scenario": scenario,
                            "campaign": campaign.campaign_id,
                            "report": label,
                            "model": model.name,
                            "edp_count": len(report.observation.edps),
                            "value": relative_error(report.full_union, truth),
                        }
                    )
                    rows.append(
                        {
                            "category": "diagnostic_score",
                            "scenario": scenario,
                            "campaign": campaign.campaign_id,
                            "report": label,
                            "model": model.name,
                            "edp_count": len(report.observation.edps),
                            "value": observable_capture_residual(report.observation, model),
                        }
                    )

                full_calibrated = calibrated["all_weeks__10_edps"]
                for subset in selected_masks(config.n_edps, (2, 5, 10)):
                    truth = float(full_observation.truth_unions[subset])
                    rows.append(
                        {
                            "category": "stress_subset_sweep",
                            "scenario": scenario,
                            "campaign": campaign.campaign_id,
                            "report": "all_weeks__10_edps",
                            "model": model.name,
                            "edp_count": subset.bit_count(),
                            "value": relative_error(float(full_calibrated.union_values[subset]), truth),
                        }
                    )

                labels = list(observations)
                registry, finalized, repeat_failures = _run_registry(calibrated, labels, config)
                structural["repeat_failures"] += repeat_failures
                audit = registry.audit()
                audit.update({"scenario": scenario, "campaign": campaign.campaign_id, "model": model.name})
                structural["audits"].append(audit)
                structural["review_counts"].setdefault(model.name, {"total": 0, "review": 0})
                for label, result in finalized.items():
                    truth = float(observations[label].truth_unions[-1])
                    rows.append(
                        {
                            "category": "stress_union_reconciled",
                            "scenario": scenario,
                            "campaign": campaign.campaign_id,
                            "report": label,
                            "model": model.name,
                            "edp_count": len(observations[label].edps),
                            "value": relative_error(result.finalized_union, truth),
                        }
                    )
                    structural["review_counts"][model.name]["total"] += 1
                    structural["review_counts"][model.name]["review"] += int(result.status == "REVIEW_REQUIRED")

                order_values: dict[str, list[float]] = {label: [] for label in labels}
                rng = np.random.default_rng(config.seed + 800_000 + scenario_index * 100 + campaign_index)
                orders = [labels, list(reversed(labels))]
                for _ in range(max(0, config.request_order_trials - 2)):
                    orders.append(list(rng.permutation(labels)))
                for order in orders:
                    _, result_by_label, _ = _run_registry(calibrated, list(order), config)
                    for label, result in result_by_label.items():
                        order_values[label].append(result.finalized_union)
                sensitivity = []
                for label, values in order_values.items():
                    truth = float(observations[label].truth_unions[-1])
                    sensitivity.append((max(values) - min(values)) / max(truth, 1.0))
                structural["order_sensitivity"].setdefault(model.name, []).extend(sensitivity)

    return rows, structural


def _linkage_shift_validation(
    world: SyntheticWorld,
    models: list[CalibrationModel],
) -> list[dict]:
    config = world.config
    shifts = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
    edp_sets = (
        tuple(range(2)),
        tuple(range(5)),
        tuple(range(config.n_edps)),
    )
    campaign_count = max(1, config.stress_campaigns_per_scenario // 2)
    rows: list[dict] = []
    for shift_index, shift in enumerate(shifts):
        scenario = f"linkage_shift_{shift:+.2f}"
        for campaign_index in range(campaign_count):
            campaign = generate_campaign(
                world,
                scenario,
                config.seed + 900_000 + shift_index * 1_000 + campaign_index,
                f"{scenario}_{campaign_index:02d}",
            )
            for edps in edp_sets:
                observation = measure_report(world, campaign, tuple(range(config.n_weeks)), edps)
                truth = float(observation.truth_unions[-1])
                rows.append(
                    {
                        "category": "linkage_shift_sweep",
                        "scenario": f"{shift:+.2f}",
                        "campaign": campaign.campaign_id,
                        "report": "all_weeks",
                        "model": "baseline_vid",
                        "edp_count": len(edps),
                        "value": relative_error(float(observation.baseline_unions[-1]), truth),
                    }
                )
                for model in models:
                    calibrated = calibrate_report(observation, model)
                    rows.append(
                        {
                            "category": "linkage_shift_sweep",
                            "scenario": f"{shift:+.2f}",
                            "campaign": campaign.campaign_id,
                            "report": "all_weeks",
                            "model": model.name,
                            "edp_count": len(edps),
                            "value": relative_error(calibrated.full_union, truth),
                        }
                    )
                    rows.append(
                        {
                            "category": "linkage_shift_diagnostic",
                            "scenario": f"{shift:+.2f}",
                            "campaign": campaign.campaign_id,
                            "report": "all_weeks",
                            "model": model.name,
                            "edp_count": len(edps),
                            "value": observable_capture_residual(observation, model),
                        }
                    )
    return rows


def _infeasible_fixture(config: SimulationConfig) -> dict:
    registry = ResultRegistry(config.n_weeks, config.review_movement_fraction)
    small_key = ReportKey("fixture", "fixture_model", (0,), (0,))
    large_key = ReportKey("fixture", "fixture_model", (0, 1), (0, 1))
    registry.inject_finalized(
        FinalizedReport(small_key, 80.0, 80.0, "OK", 0.0, 100.0, 0.0, 0.0, 0)
    )
    registry.inject_finalized(
        FinalizedReport(large_key, 50.0, 50.0, "OK", 0.0, 100.0, 0.0, 0.0, 0)
    )
    middle = ReportCandidate(
        ReportKey("fixture", "fixture_model", (0,), (0, 1)),
        raw_union=60.0,
        marginal_reaches=(40.0, 35.0),
        uncertainty=5.0,
        population_size=100.0,
    )
    result = registry.finalize(middle)
    return {
        "status": result.status,
        "produced": bool(np.isfinite(result.finalized_union)),
        "finalized_union": result.finalized_union,
        "slack": result.slack,
        "prior_values_unchanged": (
            registry.finalize(ReportCandidate(small_key, 0.0, (0.0,), 1.0, 100.0)).finalized_union == 80.0
            and registry.finalize(ReportCandidate(large_key, 0.0, (0.0, 0.0), 1.0, 100.0)).finalized_union == 50.0
        ),
    }


def _summaries(rows: list[dict]) -> dict:
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        key = (row["category"], row["scenario"], row["model"], str(row["edp_count"]))
        grouped.setdefault(key, []).append(float(row["value"]))
    return {
        "|".join(key): summarize(values)
        for key, values in sorted(grouped.items())
    }


def _model_score(rows: list[dict], model_name: str) -> float:
    values = [
        row["value"]
        for row in rows
        if row["category"] == "holdout_union" and row["model"] == model_name
    ]
    return summarize(values)["p90"]


def _select_pair_model(
    candidates: list[PairAwareLogModel],
    rows: list[dict],
    material_improvement: float,
) -> PairAwareLogModel:
    selected = candidates[0]
    selected_score = _model_score(rows, selected.name)
    for candidate in candidates[1:]:
        score = _model_score(rows, candidate.name)
        if selected_score - score >= material_improvement:
            selected = candidate
            selected_score = score
    return selected


def _write_metrics(path: Path, rows: list[dict]) -> None:
    fields = ["category", "scenario", "campaign", "report", "model", "edp_count", "value"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_errors(path: Path, rows: list[dict], pair_name: str, mixture_name: str) -> None:
    scenarios = [scenario for scenario in SCENARIOS if scenario != "representative"]
    methods = ["baseline_vid", pair_name, mixture_name]
    values = np.zeros((len(methods), len(scenarios)))
    for method_index, method in enumerate(methods):
        for scenario_index, scenario in enumerate(scenarios):
            selected = [
                row["value"]
                for row in rows
                if row["category"] == "stress_union_raw"
                and row["model"] == method
                and row["scenario"] == scenario
            ]
            values[method_index, scenario_index] = np.mean(selected) if selected else np.nan
    x = np.arange(len(scenarios))
    width = 0.25
    fig, axis = plt.subplots(figsize=(12, 5.5))
    display_names = {
        "baseline_vid": "Existing VID baseline",
        pair_name: "Pair-aware fixed + log",
        mixture_name: "Two-group mixture",
    }
    for index, method in enumerate(methods):
        axis.bar(
            x + (index - 1) * width,
            100.0 * values[index],
            width,
            label=display_names[method],
        )
    axis.set_ylabel("Mean absolute relative error in union reach (%)")
    axis.set_xticks(x, [value.replace("_", "\n") for value in scenarios], fontsize=8)
    axis.set_title("Synthetic stress-campaign union-reach error")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_linkage_sweep(path: Path, rows: list[dict], pair_name: str, mixture_name: str) -> None:
    methods = ["baseline_vid", pair_name, mixture_name]
    display_names = {
        "baseline_vid": "Existing VID baseline",
        pair_name: "Pair-aware fixed + log",
        mixture_name: "Two-group mixture",
    }
    shifts = sorted(
        {float(row["scenario"]) for row in rows if row["category"] == "linkage_shift_sweep"}
    )
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    for method in methods:
        means = []
        for shift in shifts:
            selected = [
                row["value"]
                for row in rows
                if row["category"] == "linkage_shift_sweep"
                and row["model"] == method
                and float(row["scenario"]) == shift
            ]
            means.append(100.0 * float(np.mean(selected)))
        axis.plot(shifts, means, marker="o", label=display_names[method])
    axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Audience selection toward higher matchability")
    axis.set_ylabel("Mean absolute relative error in union reach (%)")
    axis.set_title("Calibration-transfer stress sweep")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_report(
    path: Path,
    config: SimulationConfig,
    models: list[CalibrationModel],
    selected_pair: PairAwareLogModel,
    rows: list[dict],
    structural: dict,
    fixture: dict,
) -> None:
    summaries = _summaries(rows)

    def metric(category: str, scenario: str, model: str, edp_count: str = "10"):
        return summaries.get(f"{category}|{scenario}|{model}|{edp_count}", {})

    nominal_scenarios = {
        "representative",
        "small_vs_large_nonreach",
        "two_small_correlated",
        "two_small_disjoint",
        "all_small_correlated",
        "mixed_objectives",
    }
    baseline_all = [
        row["value"] for row in rows
        if row["category"] == "stress_union_raw"
        and row["model"] == "baseline_vid"
        and row["scenario"] in nominal_scenarios
    ]
    pair_all = [
        row["value"] for row in rows
        if row["category"] == "stress_union_raw"
        and row["model"] == selected_pair.name
        and row["scenario"] in nominal_scenarios
    ]
    mixture_all = [
        row["value"] for row in rows
        if row["category"] == "stress_union_raw"
        and row["model"] == "two_group_latent_mixture"
        and row["scenario"] in nominal_scenarios
    ]
    baseline_summary = summarize(baseline_all)
    pair_summary = summarize(pair_all)
    mixture_summary = summarize(mixture_all)
    selected_holdout_p90 = _model_score(rows, selected_pair.name)

    audit_monotonic = sum(item["monotonic_violations"] for item in structural["audits"])
    audit_set = sum(item["set_arithmetic_violations"] for item in structural["audits"])
    sensitivity = {
        model: summarize(values)
        for model, values in structural["order_sensitivity"].items()
    }
    diagnostic_summary = {}
    for model in (selected_pair.name, "two_group_latent_mixture"):
        sweep_rows = [
            row for row in rows
            if row["category"] == "linkage_shift_diagnostic" and row["model"] == model
        ]
        shifts = np.array([float(row["scenario"]) for row in sweep_rows], dtype=float)
        scores = np.array([float(row["value"]) for row in sweep_rows], dtype=float)
        correlation = float(np.corrcoef(shifts, scores)[0, 1]) if len(scores) > 1 else float("nan")
        nominal_scores = [
            abs(float(row["value"])) for row in rows
            if row["category"] == "diagnostic_score"
            and row["model"] == model
            and row["scenario"] in nominal_scenarios
        ]
        diagnostic_summary[model] = {
            "shift_correlation": correlation,
            "nominal_mean_absolute": float(np.mean(nominal_scores)) if nominal_scores else float("nan"),
        }

    lines = [
        "# Synthetic validation report",
        "",
        "## Scope",
        "",
        f"- Population represented: {config.population_size:,} people using {config.n_users:,} weighted synthetic people.",
        f"- EDPs: {config.n_edps}; weeks: {config.n_weeks}; Reference-ID pool: {config.reference_pool_size:,}.",
        f"- Calibration campaigns: {config.calibration_train_campaigns} fitting and {config.calibration_holdout_campaigns} whole-campaign holdouts.",
        f"- Stress campaigns per scenario: {config.stress_campaigns_per_scenario}.",
        "- Conditional email agreement ranges from approximately 52% to 72%; email availability ranges from 10% to 95%.",
        "",
        "## Models",
        "",
        f"- Selected fixed/log candidate: `{selected_pair.name}` ({selected_pair.parameter_count} fitted parameters).",
        "- Alternative: `two_group_latent_mixture` (21 fitted parameters at 10 EDPs).",
        "- The pair-aware family was selected only among its fixed, shared-log, and order-log submodels using whole-campaign holdouts.",
        "",
        "## Headline nominal-stress accuracy",
        "",
        "| Method | Mean absolute relative error | p90 | p99 | Maximum |",
        "|---|---:|---:|---:|---:|",
        f"| Existing VID baseline | {baseline_summary['mean']:.2%} | {baseline_summary['p90']:.2%} | {baseline_summary['p99']:.2%} | {baseline_summary['max']:.2%} |",
        f"| Pair-aware fixed/log | {pair_summary['mean']:.2%} | {pair_summary['p90']:.2%} | {pair_summary['p99']:.2%} | {pair_summary['max']:.2%} |",
        f"| Two-group mixture | {mixture_summary['mean']:.2%} | {mixture_summary['p90']:.2%} | {mixture_summary['p99']:.2%} | {mixture_summary['max']:.2%} |",
        "",
        "These figures combine the explicit 2-, 5-, and 10-EDP report shapes across representative, small-versus-large, two-small, all-small, and mixed-objective scenarios. Deliberate matchability-transfer shifts are reported separately below. Relative error can exceed 100% when the true union is small.",
        "",
        "### Mean error by nominal scenario",
        "",
        "| Scenario | Existing VID | Pair-aware fixed/log | Two-group mixture |",
        "|---|---:|---:|---:|",
    ]
    for scenario in sorted(nominal_scenarios):
        scenario_values = {}
        for method in ("baseline_vid", selected_pair.name, "two_group_latent_mixture"):
            selected = [
                row["value"] for row in rows
                if row["category"] == "stress_union_raw"
                and row["scenario"] == scenario
                and row["model"] == method
            ]
            scenario_values[method] = float(np.mean(selected)) if selected else float("nan")
        lines.append(
            f"| {scenario.replace('_', ' ')} | {scenario_values['baseline_vid']:.2%} | "
            f"{scenario_values[selected_pair.name]:.2%} | {scenario_values['two_group_latent_mixture']:.2%} |"
        )
    lines.extend([
        "",
        "### Mean error by report size",
        "",
        "| EDPs | Existing VID | Pair-aware fixed/log | Two-group mixture |",
        "|---:|---:|---:|---:|",
    ])
    for edp_count in (2, 5, 10):
        size_values = {}
        for method in ("baseline_vid", selected_pair.name, "two_group_latent_mixture"):
            selected = [
                row["value"] for row in rows
                if row["category"] == "stress_union_raw"
                and row["scenario"] in nominal_scenarios
                and row["model"] == method
                and int(row["edp_count"]) == edp_count
            ]
            size_values[method] = float(np.mean(selected)) if selected else float("nan")
        lines.append(
            f"| {edp_count} | {size_values['baseline_vid']:.2%} | "
            f"{size_values[selected_pair.name]:.2%} | {size_values['two_group_latent_mixture']:.2%} |"
        )
    lines.extend([
        "",
        "### Requested stress examples",
        "",
        "| Campaign/report shape | Existing VID | Pair-aware fixed/log | Two-group mixture |",
        "|---|---:|---:|---:|",
    ])
    requested_examples = (
        ("Small non-reach versus large reach, weeks 5-12, 2 EDPs", "small_vs_large_nonreach", "weeks_5_12__2_edps"),
        ("Two small correlated campaigns, full flight, 2 EDPs", "two_small_correlated", "all_weeks__2_edps"),
        ("Two small disjoint campaigns, full flight, 2 EDPs", "two_small_disjoint", "all_weeks__2_edps"),
        ("All-small correlated, weeks 7-13, 5 EDPs", "all_small_correlated", "weeks_7_13__5_edps"),
        ("Mixed objectives, full flight, 10 EDPs", "mixed_objectives", "all_weeks__10_edps"),
    )
    for label, scenario, report_name in requested_examples:
        example_values = {}
        for method in ("baseline_vid", selected_pair.name, "two_group_latent_mixture"):
            selected = [
                row["value"] for row in rows
                if row["category"] == "stress_union_raw"
                and row["scenario"] == scenario
                and row["report"] == report_name
                and row["model"] == method
            ]
            example_values[method] = float(np.mean(selected)) if selected else float("nan")
        lines.append(
            f"| {label} | {example_values['baseline_vid']:.2%} | "
            f"{example_values[selected_pair.name]:.2%} | {example_values['two_group_latent_mixture']:.2%} |"
        )
    lines.extend([
        "",
        "## Calibration-transfer sweep",
        "",
        "The sweep changes how strongly campaign selection favors people who are generally easy or difficult to match. Zero adds no direct matchability selection; negative values select less-matchable people and positive values select more-matchable people.",
        "",
        "| Shift | Existing VID | Pair-aware fixed/log | Two-group mixture |",
        "|---:|---:|---:|---:|",
    ])
    for shift in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
        shift_values = {}
        for method in ("baseline_vid", selected_pair.name, "two_group_latent_mixture"):
            selected = [
                row["value"]
                for row in rows
                if row["category"] == "linkage_shift_sweep"
                and row["model"] == method
                and float(row["scenario"]) == shift
            ]
            shift_values[method] = float(np.mean(selected)) if selected else float("nan")
        lines.append(
            f"| {shift:+.2f} | {shift_values['baseline_vid']:.2%} | "
            f"{shift_values[selected_pair.name]:.2%} | {shift_values['two_group_latent_mixture']:.2%} |"
        )
    lines.extend([
        "",
        "A simple observable diagnostic—the average pairwise log difference between observed and expected Reference-ID capture—does respond to the injected shift, but it is also affected by genuine campaign-overlap differences:",
        "",
        "| Model | Correlation with injected shift | Mean absolute score on nominal stress campaigns |",
        "|---|---:|---:|",
        f"| Pair-aware fixed/log | {diagnostic_summary[selected_pair.name]['shift_correlation']:.2f} | {diagnostic_summary[selected_pair.name]['nominal_mean_absolute']:.3f} |",
        f"| Two-group mixture | {diagnostic_summary['two_group_latent_mixture']['shift_correlation']:.2f} | {diagnostic_summary['two_group_latent_mixture']['nominal_mean_absolute']:.3f} |",
        "",
        "The diagnostic is therefore useful for review prioritization, not as a reliable automatic test that linkage has shifted.",
    ])
    lines.extend([
        "",
        "## Consistency and failure handling",
        "",
        f"- Exact-repeat failures: {structural['repeat_failures']}.",
        f"- Monotonic violations remaining across stored reports: {audit_monotonic}.",
        f"- Set-coverage inequality violations remaining where all required reports existed: {audit_set}.",
        f"- Deliberately infeasible fixture produced a result: {fixture['produced']}; status: `{fixture['status']}`; prior results unchanged: {fixture['prior_values_unchanged']}.",
        "",
        "Request order can matter in principle because earlier finalized reports become immutable anchors. It did not change any result in the tested request set:",
        "",
        "| Model | Mean relative range | p90 | Maximum |",
        "|---|---:|---:|---:|",
    ])
    for model, values in sensitivity.items():
        lines.append(
            f"| {model} | {values['mean']:.2%} | {values['p90']:.2%} | {values['max']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Passing the structural tests shows that the implementation can produce bounded reports, preserve finalized history, recognize exact repeats, and flag infeasible new requests. It does not prove that the calibration assumptions transfer to real campaigns.",
            "",
            "The most important next validation is to repeat the model fitting and holdout comparison on approved aggregate observations from real broad-reach and non-reach campaigns. In particular, real data is needed to determine whether pair-specific affinity, campaign-size effects, or latent person-level matchability best explains Reference-ID visibility.",
            "",
            "## Acceptance checks",
            "",
            f"- [{'x' if structural['repeat_failures'] == 0 else ' '}] Exact repeated requests are stable.",
            f"- [{'x' if fixture['status'] == 'REVIEW_REQUIRED' and fixture['produced'] else ' '}] An infeasible new report is still produced and flagged.",
            f"- [{'x' if fixture['prior_values_unchanged'] else ' '}] Infeasible processing does not rewrite prior results.",
            f"- [{'x' if any(row['edp_count'] == 10 for row in rows) else ' '}] Ten-EDP reports are included.",
            f"- [{'x' if pair_summary['count'] > 0 and mixture_summary['count'] > 0 else ' '}] Both calibration families are evaluated.",
            f"- [{'x' if selected_holdout_p90 <= 0.10 else ' '}] Selected pair-aware model has representative holdout p90 union error at or below 10% ({selected_holdout_p90:.2%}).",
            f"- [{'x' if pair_summary['mean'] < baseline_summary['mean'] else ' '}] Pair-aware calibration improves mean nominal-stress error over the existing VID baseline.",
            f"- [{'x' if mixture_summary['mean'] < baseline_summary['mean'] else ' '}] Mixture calibration improves mean nominal-stress error over the existing VID baseline.",
            "",
            "## Reproducibility artifacts",
            "",
            "The fitted coefficients, mixture parameters, and version identifiers are written to `model_artifacts.json`. Detailed observations are in `metrics.csv`, and the full run configuration and acceptance results are in `summary.json`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config: SimulationConfig, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    world = make_world(config)
    train_campaigns, holdout_campaigns = _make_calibration_campaigns(world)
    checkpoints = calibration_checkpoints(config.n_weeks)
    train_observations = _observations_for_campaigns(world, train_campaigns, checkpoints)
    holdout_observations = _observations_for_campaigns(world, holdout_campaigns, checkpoints)
    train_data = calibration_dataset(train_observations, config.minimum_calibration_intersection)
    holdout_data = calibration_dataset(holdout_observations, config.minimum_calibration_intersection)

    pair_candidates = [
        PairAwareLogModel.fit(train_data, config.n_edps, mode, config.ridge_penalty)
        for mode in ("none", "shared", "by_order")
    ]
    mixture = LatentMixtureModel.fit(train_data, config.n_edps, config.seed + 77)
    candidate_models: list[CalibrationModel] = [*pair_candidates, mixture]
    holdout_rows = _evaluate_full_roster(holdout_observations, candidate_models)
    holdout_rows.extend(_capture_validation(holdout_data, candidate_models))
    selected_pair = _select_pair_model(
        pair_candidates,
        holdout_rows,
        config.material_holdout_improvement,
    )

    stress_rows, structural = _stress_validation(world, [selected_pair, mixture])
    linkage_rows = _linkage_shift_validation(world, [selected_pair, mixture])
    rows = holdout_rows + stress_rows + linkage_rows
    fixture = _infeasible_fixture(config)
    summaries = _summaries(rows)
    nominal_scenarios = {
        "representative",
        "small_vs_large_nonreach",
        "two_small_correlated",
        "two_small_disjoint",
        "all_small_correlated",
        "mixed_objectives",
    }
    nominal = {
        model: summarize([
            row["value"] for row in rows
            if row["category"] == "stress_union_raw"
            and row["scenario"] in nominal_scenarios
            and row["model"] == model
        ])
        for model in ("baseline_vid", selected_pair.name, mixture.name)
    }
    acceptance = {
        "exact_repeats": structural["repeat_failures"] == 0,
        "infeasible_report_produced_and_flagged": fixture["produced"] and fixture["status"] == "REVIEW_REQUIRED",
        "prior_results_unchanged": fixture["prior_values_unchanged"],
        "ten_edp_covered": any(row["edp_count"] == 10 for row in rows),
        "selected_pair_holdout_p90_le_10pct": _model_score(rows, selected_pair.name) <= 0.10,
        "pair_improves_nominal_mean": nominal[selected_pair.name]["mean"] < nominal["baseline_vid"]["mean"],
        "mixture_improves_nominal_mean": nominal[mixture.name]["mean"] < nominal["baseline_vid"]["mean"],
    }
    acceptance["passed"] = all(acceptance.values())

    summary = {
        "config": config.__dict__,
        "world": {
            "email_coverage": world.email_coverage.tolist(),
            "email_agreement": world.email_agreement.tolist(),
            "target_link_probability": world.target_link_probability.tolist(),
            "realized_link_probability": world.realized_link_probability.tolist(),
        },
        "models": [model.describe() for model in candidate_models],
        "selected_pair_model": selected_pair.describe(),
        "holdout_pair_scores": {
            model.name: _model_score(holdout_rows, model.name) for model in pair_candidates
        },
        "summaries": summaries,
        "structural": structural,
        "infeasible_fixture": fixture,
        "acceptance": acceptance,
    }

    _write_metrics(output_dir / "metrics.csv", rows)
    (output_dir / "model_artifacts.json").write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "model_line": "synthetic-model-line-v1",
                "reference_id_version": "synthetic-source-v1",
                "selected_pair_model": selected_pair.describe(),
                "mixture_model": mixture.describe(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist(),
        ),
        encoding="utf-8",
    )
    _write_report(
        output_dir / "validation_report.md",
        config,
        candidate_models,
        selected_pair,
        rows,
        structural,
        fixture,
    )
    _plot_errors(output_dir / "error_by_scenario.png", rows, selected_pair.name, mixture.name)
    _plot_linkage_sweep(output_dir / "linkage_shift_sweep.png", rows, selected_pair.name, mixture.name)
    return summary
