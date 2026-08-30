from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .evaluation import CalibratedReport, calibrate_report, relative_error, summarize
from .experiment import calibration_checkpoints
from .joint_decoding import (
    JointDecoderConfig,
    calibrate_report_joint,
    calibrate_report_pairwise_maximum_entropy,
)
from .measurement import calibration_dataset, measure_report
from .models import CalibrationModel, LatentMixtureModel, PairAwareLogModel
from .population import META_CAMPAIGN_SCENARIOS, generate_campaign, make_world
from .research_models import (
    DirectPairLogModel,
    HierarchicalPairModel,
    LowRankAffinityModel,
    MonotoneSplineCaptureModel,
    MultiGroupMixtureModel,
    fit_low_rank_pair_affinity,
)
from .sets import intersection_matrix


@dataclass(frozen=True)
class BenchmarkMethod:
    name: str
    label: str
    category: str
    run: Callable
    parameter_count: int | str
    explanation: str


def _calibration_observations(world, campaigns):
    all_edps = tuple(range(world.config.n_edps))
    return [
        measure_report(world, campaign, weeks, all_edps)
        for campaign in campaigns
        for weeks in calibration_checkpoints(world.config.n_weeks)
    ]


def _fit_models(world):
    config = world.config
    total = config.calibration_train_campaigns + config.calibration_holdout_campaigns
    campaigns = [
        generate_campaign(
            world,
            "representative",
            config.seed + 10_000 + index,
            f"benchmark_calibration_{index:03d}",
        )
        for index in range(total)
    ]
    train_campaigns = campaigns[: config.calibration_train_campaigns]
    holdout_campaigns = campaigns[config.calibration_train_campaigns :]
    train_observations = _calibration_observations(world, train_campaigns)
    train_data = calibration_dataset(
        train_observations,
        config.minimum_calibration_intersection,
    )

    pair_fixed = PairAwareLogModel.fit(
        train_data,
        config.n_edps,
        "none",
        config.ridge_penalty,
    )
    pair_shared = PairAwareLogModel.fit(
        train_data,
        config.n_edps,
        "shared",
        config.ridge_penalty,
    )
    pair_order = PairAwareLogModel.fit(
        train_data,
        config.n_edps,
        "by_order",
        config.ridge_penalty,
    )
    direct_pair = DirectPairLogModel.fit(
        train_data,
        config.n_edps,
        config.ridge_penalty,
    )
    hierarchical = HierarchicalPairModel.fit(
        train_data,
        config.n_edps,
        "shared",
        1.0,
    )
    spline = MonotoneSplineCaptureModel.fit(
        train_data,
        config.n_edps,
        "decreasing",
        config.ridge_penalty,
    )
    low_rank = LowRankAffinityModel.fit(
        train_data,
        config.n_edps,
        rank=2,
        ridge_penalty=0.05,
        seed=config.seed + 501,
    )
    mixture_two = LatentMixtureModel.fit(
        train_data,
        config.n_edps,
        config.seed + 502,
    )
    mixture_three = MultiGroupMixtureModel.fit(
        train_data,
        config.n_edps,
        n_groups=3,
        seed=config.seed + 503,
    )
    pair_affinity = fit_low_rank_pair_affinity(
        train_data,
        mixture_two,
        config.n_edps,
        rank=2,
    )
    return {
        "train_data": train_data,
        "holdout_campaigns": holdout_campaigns,
        "pair_fixed": pair_fixed,
        "pair_shared": pair_shared,
        "pair_order": pair_order,
        "direct_pair": direct_pair,
        "hierarchical": hierarchical,
        "spline": spline,
        "low_rank": low_rank,
        "mixture_two": mixture_two,
        "mixture_three": mixture_three,
        "pair_affinity": pair_affinity,
    }


def _method_roster(fitted) -> list[BenchmarkMethod]:
    pair_fixed = fitted["pair_fixed"]
    pair_shared = fitted["pair_shared"]
    pair_order = fitted["pair_order"]
    direct_pair = fitted["direct_pair"]
    hierarchical = fitted["hierarchical"]
    spline = fitted["spline"]
    low_rank = fitted["low_rank"]
    mixture_two = fitted["mixture_two"]
    mixture_three = fitted["mixture_three"]
    affinity = fitted["pair_affinity"]

    def divided(model: CalibrationModel):
        return lambda observation: calibrate_report(observation, model)

    return [
        BenchmarkMethod(
            "baseline_vid",
            "Existing VID",
            "baseline",
            lambda observation: None,
            0,
            "Population-rate overlap using the per-EDP VID reaches.",
        ),
        BenchmarkMethod(
            "pair_fixed_divide",
            "Pair-aware fixed",
            "capture curve",
            divided(pair_fixed),
            pair_fixed.parameter_count,
            "Constant bounded capture rates with pair-specific effects.",
        ),
        BenchmarkMethod(
            "pair_log_shared_divide",
            "Pair-aware fixed + shared log",
            "capture curve",
            divided(pair_shared),
            pair_shared.parameter_count,
            "Pair-specific effects plus one campaign-size slope.",
        ),
        BenchmarkMethod(
            "pair_log_order_divide",
            "Pair-aware order-specific log",
            "capture curve",
            divided(pair_order),
            pair_order.parameter_count,
            "A separate campaign-size slope for every intersection order.",
        ),
        BenchmarkMethod(
            "shape_spline_divide",
            "Shape-constrained scale spline",
            "capture curve",
            divided(spline),
            spline.parameter_count,
            "A smooth monotone campaign-size response instead of one log slope.",
        ),
        BenchmarkMethod(
            "hierarchical_pair_divide",
            "Hierarchical EDP + pair effects",
            "capture curve",
            divided(hierarchical),
            hierarchical.parameter_count,
            "EDP main effects plus strongly pooled pair residuals.",
        ),
        BenchmarkMethod(
            "low_rank_divide",
            "Low-rank EDP affinity",
            "capture curve",
            divided(low_rank),
            low_rank.parameter_count,
            "Two shared dimensions represent recurring EDP affinity patterns.",
        ),
        BenchmarkMethod(
            "two_group_divide",
            "Two-group mixture",
            "capture mixture",
            divided(mixture_two),
            mixture_two.parameter_count,
            "Two fixed person-level matchability groups.",
        ),
        BenchmarkMethod(
            "three_group_divide",
            "Three-group mixture",
            "capture mixture",
            divided(mixture_three),
            mixture_three.parameter_count,
            "Adds a middle matchability group to the two-group model.",
        ),
        BenchmarkMethod(
            "pairwise_maxent_logit",
            "Pairwise inference: logit calibration",
            "pairwise inference",
            lambda observation: calibrate_report_pairwise_maximum_entropy(
                observation,
                pair_shared,
                pair_ridge=1e-6,
                evidence_half_saturation=0.1,
                name="pairwise_maxent_logit",
            ),
            f"{pair_shared.parameter_count} calibration + 55 report",
            "Calibrate only the 45 pairs, then infer all higher orders by maximum entropy.",
        ),
        BenchmarkMethod(
            "pairwise_maxent_direct",
            "Pairwise inference: direct calibration",
            "pairwise inference",
            lambda observation: calibrate_report_pairwise_maximum_entropy(
                observation,
                direct_pair,
                pair_ridge=1e-6,
                evidence_half_saturation=0.1,
                name="pairwise_maxent_direct",
            ),
            f"{direct_pair.parameter_count} calibration + 55 report",
            "Uses bounded c=a+b log(size) pair calibration before maximum-entropy inference.",
        ),
        BenchmarkMethod(
            "pairwise_maxent_mixture",
            "Pairwise inference: two-group mixture",
            "pairwise inference",
            lambda observation: calibrate_report_pairwise_maximum_entropy(
                observation,
                mixture_two,
                pair_ridge=1e-6,
                evidence_half_saturation=0.1,
                name="pairwise_maxent_mixture",
            ),
            f"{mixture_two.parameter_count} calibration + 55 report",
            "Uses the two-group mixture to calibrate pairs, then infers higher orders by maximum entropy.",
        ),
        BenchmarkMethod(
            "oracle_pairwise_maxent",
            "Oracle pairwise closure (diagnostic)",
            "diagnostic oracle",
            lambda observation: calibrate_report_pairwise_maximum_entropy(
                observation,
                pair_shared,
                pair_ridge=1e-6,
                evidence_half_saturation=0.1,
                name="oracle_pairwise_maxent",
                pair_target_intersections=observation.truth_intersections,
            ),
            "Unavailable in production",
            "Uses true synthetic pairs to isolate the irreducible error from inferring higher orders from pairs.",
        ),
        BenchmarkMethod(
            "joint_two_group_exact",
            "Joint two-group pattern decoder",
            "joint decoder",
            lambda observation: calibrate_report_joint(
                observation,
                mixture_two,
                JointDecoderConfig(
                    "joint_two_group_exact",
                    response_mode="mixture_exact",
                    prior_strength=1e-4,
                    evidence_half_saturation=5.0,
                ),
            ),
            mixture_two.parameter_count,
            "Fits all true Venn cells directly to the observed multi-EDP Reference-ID patterns.",
        ),
        BenchmarkMethod(
            "joint_three_group_exact",
            "Joint three-group pattern decoder",
            "joint decoder",
            lambda observation: calibrate_report_joint(
                observation,
                mixture_three,
                JointDecoderConfig(
                    "joint_three_group_exact",
                    response_mode="mixture_exact",
                    prior_strength=1e-4,
                    evidence_half_saturation=5.0,
                ),
            ),
            mixture_three.parameter_count,
            "Joint pattern decoding with three person-level matchability groups.",
        ),
        BenchmarkMethod(
            "joint_low_rank_inclusive",
            "Joint low-rank inclusive decoder",
            "joint decoder",
            lambda observation: calibrate_report_joint(
                observation,
                low_rank,
                JointDecoderConfig(
                    "joint_low_rank_inclusive",
                    response_mode="inclusive",
                    prior_strength=1e-4,
                    evidence_half_saturation=5.0,
                ),
            ),
            low_rank.parameter_count,
            "Fits one audience table to all inclusive overlaps using low-rank capture rates.",
        ),
        BenchmarkMethod(
            "joint_mixture_affinity",
            "Joint mixture + low-rank affinity",
            "joint decoder",
            lambda observation: calibrate_report_joint(
                observation,
                mixture_two,
                JointDecoderConfig(
                    "joint_mixture_affinity",
                    response_mode="mixture_exact",
                    prior_strength=1e-4,
                    evidence_half_saturation=5.0,
                ),
                affinity_matrix=affinity,
            ),
            f"{mixture_two.parameter_count} + rank-2 affinity",
            "Adds repeatable EDP-pair residual affinity to the joint mixture response.",
        ),
        BenchmarkMethod(
            "bayesian_map_affinity",
            "Bayesian/MAP joint affinity",
            "Bayesian benchmark",
            lambda observation: calibrate_report_joint(
                observation,
                mixture_two,
                JointDecoderConfig(
                    "bayesian_map_affinity",
                    response_mode="mixture_exact",
                    prior_strength=0.01,
                    evidence_half_saturation=5.0,
                    map_iterations=4,
                ),
                affinity_matrix=affinity,
            ),
            f"{mixture_two.parameter_count} + rank-2 affinity + cell prior",
            "Poisson-IRLS MAP approximation with a weak prior around the baseline audience table.",
        ),
    ]


def _report_shapes(n_edps: int, n_weeks: int):
    return (
        ("full_2", tuple(range(n_weeks)), tuple(range(2))),
        ("full_5", tuple(range(n_weeks)), tuple(range(5))),
        ("full_10", tuple(range(n_weeks)), tuple(range(n_edps))),
        ("weeks_1_3__10", tuple(range(min(3, n_weeks))), tuple(range(n_edps))),
        ("weeks_5_12__2", tuple(range(4, min(12, n_weeks))), tuple(range(2))),
        ("weeks_7_13__5", tuple(range(6, n_weeks)), tuple(range(5))),
    )


def _final_intersections(observation, result):
    if result is None:
        return observation.baseline_intersections
    output = np.zeros(len(observation.global_masks), dtype=float)
    output[1:] = intersection_matrix(len(observation.edps)) @ result.exclusive_cells[1:]
    return output


def _evaluate_observation(observation, methods, phase, rows):
    for method in methods:
        start = perf_counter()
        result = method.run(observation)
        elapsed = perf_counter() - start
        union_estimate = (
            float(observation.baseline_unions[-1])
            if result is None
            else float(result.full_union)
        )
        truth_union = float(observation.truth_unions[-1])
        rows.append(
            {
                "phase": phase,
                "metric": "union_error",
                "scenario": observation.scenario,
                "campaign": observation.campaign_id,
                "report": f"weeks_{len(observation.weeks)}__edps_{len(observation.edps)}",
                "method": method.name,
                "method_label": method.label,
                "method_category": method.category,
                "edp_count": len(observation.edps),
                "intersection_order": 0,
                "value": relative_error(union_estimate, truth_union),
                "signed_value": (union_estimate - truth_union) / max(truth_union, 1.0),
                "runtime_seconds": elapsed,
            }
        )
        estimated_intersections = _final_intersections(observation, result)
        for order_label, predicate in (
            (2, lambda order: order == 2),
            (3, lambda order: order == 3),
            (4, lambda order: order == 4),
            (5, lambda order: order >= 5),
        ):
            absolute_error = 0.0
            signed_error = 0.0
            truth_total = 0.0
            for mask in range(1, len(observation.global_masks)):
                order = mask.bit_count()
                truth = float(observation.truth_intersections[mask])
                if not predicate(order) or truth < observation.person_weight:
                    continue
                estimate = float(estimated_intersections[mask])
                absolute_error += abs(estimate - truth)
                signed_error += estimate - truth
                truth_total += truth
            if truth_total > 0:
                rows.append(
                    {
                        "phase": phase,
                        "metric": "intersection_error",
                        "scenario": observation.scenario,
                        "campaign": observation.campaign_id,
                        "report": f"weeks_{len(observation.weeks)}__edps_{len(observation.edps)}",
                        "method": method.name,
                        "method_label": method.label,
                        "method_category": method.category,
                        "edp_count": len(observation.edps),
                        "intersection_order": order_label,
                        "value": absolute_error / truth_total,
                        "signed_value": signed_error / truth_total,
                        "runtime_seconds": elapsed,
                    }
                )


def _write_csv(path: Path, rows: list[dict]):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows, methods):
    output = {
        "methods": [],
        "union_by_edp_count": {},
        "intersection_by_order": {},
        "union_by_scenario": {},
    }
    for method in methods:
        method_union = [
            row["value"] for row in rows
            if row["phase"] == "stress"
            and row["metric"] == "union_error"
            and row["method"] == method.name
        ]
        method_runtime = [
            row["runtime_seconds"] for row in rows
            if row["phase"] == "stress"
            and row["metric"] == "union_error"
            and row["method"] == method.name
        ]
        output["methods"].append(
            {
                "name": method.name,
                "label": method.label,
                "category": method.category,
                "parameter_count": method.parameter_count,
                "explanation": method.explanation,
                "holdout_union": summarize([
                    row["value"] for row in rows
                    if row["phase"] == "holdout"
                    and row["metric"] == "union_error"
                    and row["method"] == method.name
                ]),
                "stress_union": summarize(method_union),
                "mean_runtime_seconds": float(np.mean(method_runtime)),
            }
        )
        for edp_count in (2, 5, 10):
            values = [
                row["value"] for row in rows
                if row["phase"] == "stress"
                and row["metric"] == "union_error"
                and row["method"] == method.name
                and row["edp_count"] == edp_count
            ]
            output["union_by_edp_count"].setdefault(str(edp_count), {})[method.name] = summarize(values)
        for order in (2, 3, 4, 5):
            values = [
                row["value"] for row in rows
                if row["phase"] == "stress"
                and row["metric"] == "intersection_error"
                and row["method"] == method.name
                and row["intersection_order"] == order
            ]
            output["intersection_by_order"].setdefault(str(order), {})[method.name] = summarize(values)
        for scenario in META_CAMPAIGN_SCENARIOS:
            values = [
                row["value"] for row in rows
                if row["phase"] == "stress"
                and row["metric"] == "union_error"
                and row["method"] == method.name
                and row["scenario"] == scenario
            ]
            output["union_by_scenario"].setdefault(scenario, {})[method.name] = summarize(values)
    return output


def _plot_union(path: Path, summary: dict):
    methods = summary["methods"]
    names = [method["label"] for method in methods]
    edp_counts = (2, 5, 10)
    colors = ("#4c78a8", "#f58518", "#54a24b")
    y = np.arange(len(methods))
    height = 0.24
    fig, axis = plt.subplots(figsize=(13, 10))
    for index, (edp_count, color) in enumerate(zip(edp_counts, colors)):
        values = [
            100.0 * summary["union_by_edp_count"][str(edp_count)][method["name"]]["mean"]
            for method in methods
        ]
        axis.barh(y + (index - 1) * height, values, height, label=f"{edp_count} EDPs", color=color)
    axis.set_yticks(y, names)
    axis.invert_yaxis()
    axis.set_xlabel("Mean absolute union-reach error (%)")
    axis.set_title("Calibration-method benchmark by report size")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_intersections(path: Path, summary: dict):
    methods = summary["methods"]
    names = [method["label"] for method in methods]
    orders = (2, 3, 4, 5)
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    y = np.arange(len(methods))
    height = 0.18
    fig, axis = plt.subplots(figsize=(13, 11))
    for index, (order, color) in enumerate(zip(orders, colors)):
        values = [
            100.0 * summary["intersection_by_order"][str(order)][method["name"]]["mean"]
            for method in methods
        ]
        label = "5+ way" if order == 5 else f"{order}-way"
        axis.barh(y + (index - 1.5) * height, values, height, label=label, color=color)
    axis.set_yticks(y, names)
    axis.invert_yaxis()
    axis.set_xlabel("Mean absolute intersection error (%)")
    axis.set_title("Where each method gains or loses accuracy")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_method_benchmark(
    output_dir: Path,
    profile: str = "quick",
    campaigns_per_scenario: int | None = None,
):
    config = SimulationConfig.for_profile(profile)
    if campaigns_per_scenario is None:
        campaigns_per_scenario = 1 if profile == "quick" else 3
    output_dir.mkdir(parents=True, exist_ok=True)
    world = make_world(config)
    fitted = _fit_models(world)
    methods = _method_roster(fitted)
    rows: list[dict] = []

    # Whole campaigns, not snapshots from a fitting campaign, are reserved for
    # the representative-population holdout evaluation.
    for campaign in fitted["holdout_campaigns"]:
        for _, weeks, edps in _report_shapes(config.n_edps, config.n_weeks)[:3]:
            observation = measure_report(world, campaign, weeks, edps)
            _evaluate_observation(observation, methods, "holdout", rows)

    for scenario_index, scenario in enumerate(META_CAMPAIGN_SCENARIOS):
        for replicate in range(campaigns_per_scenario):
            campaign = generate_campaign(
                world,
                scenario,
                config.seed + 7_000_000 + scenario_index * 10_000 + replicate,
                f"method_{scenario}_{replicate:02d}",
            )
            for _, weeks, edps in _report_shapes(config.n_edps, config.n_weeks):
                observation = measure_report(world, campaign, weeks, edps)
                _evaluate_observation(observation, methods, "stress", rows)

    summary = _summary(rows, methods)
    summary.update(
        {
            "profile": profile,
            "config": config.__dict__,
            "campaigns_per_scenario": campaigns_per_scenario,
            "method_count": len(methods),
            "scenario_count": len(META_CAMPAIGN_SCENARIOS),
            "notes": [
                "The Bayesian result is a deterministic Poisson-IRLS MAP approximation, not MCMC.",
                "Pairwise maximum entropy infers higher orders but cannot identify higher-order interactions absent from the pairs.",
                "Synthetic errors are not production accuracy estimates.",
            ],
        }
    )
    _write_csv(output_dir / "method_metrics.csv", rows)
    (output_dir / "method_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "method_artifacts.json").write_text(
        json.dumps(
            {
                key: value.describe()
                for key, value in fitted.items()
                if isinstance(value, CalibrationModel)
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_union(output_dir / "union_error_by_method.png", summary)
    _plot_intersections(output_dir / "intersection_error_by_method.png", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Reference-ID calibration methods.")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--campaigns-per-scenario", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/method_benchmark"))
    arguments = parser.parse_args()
    result = run_method_benchmark(
        arguments.output_dir,
        profile=arguments.profile,
        campaigns_per_scenario=arguments.campaigns_per_scenario,
    )
    print(json.dumps({
        "method_count": result["method_count"],
        "scenario_count": result["scenario_count"],
        "output_dir": str(arguments.output_dir),
    }, indent=2))
