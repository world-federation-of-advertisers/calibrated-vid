from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .evaluation import relative_error, summarize
from .experiment import calibration_checkpoints
from .joint_decoding import calibrate_report_pairwise_maximum_entropy
from .measurement import calibration_dataset, measure_report
from .models import LatentMixtureModel
from .population import META_CAMPAIGN_SCENARIOS, generate_campaign, make_world
from .provider_model import (
    ContextualDemographicAllocator,
    FixedDemographicAllocator,
    PanelTotalReachModel,
    ProportionalDemographicAllocator,
    demographic_distribution_error,
    demographic_reach_error,
)
from .research_models import DirectPairLogModel


SCENARIO_LABELS = {
    "broad_awareness_control": "Broad awareness / reach",
    "traffic_optimization": "Traffic optimization",
    "video_engagement_retargeting": "Engagement retargeting",
    "lead_generation": "Lead generation",
    "sales_prospecting": "Sales prospecting",
    "website_retargeting": "Website retargeting",
    "crm_customer_list": "Customer-list retargeting",
    "catalog_retargeting": "Catalog retargeting",
    "lookalike_prospecting": "Lookalike prospecting",
    "advantage_audience_expansion": "Audience expansion",
    "app_activity_retargeting": "App retargeting",
    "unrelated_niche_control": "Unrelated niche audiences",
    "mixed_funnel_portfolio": "Mixed-funnel portfolio",
}


def provider_report_specs(n_edps: int, n_weeks: int):
    all_edps = tuple(range(n_edps))
    five = tuple(range(min(5, n_edps)))
    return (
        ("weeks_1_3__2_edps", tuple(range(min(3, n_weeks))), (0, 1)),
        ("weeks_5_12__2_edps", tuple(range(4, min(12, n_weeks))), (0, 1)),
        ("all_weeks__2_edps", tuple(range(n_weeks)), (0, 1)),
        ("weeks_7_13__5_edps", tuple(range(6, n_weeks)), five),
        ("all_weeks__5_edps", tuple(range(n_weeks)), five),
        (
            "noncontiguous__5_edps",
            tuple(index for index in (0, 2, 4, 7, 10, 12) if index < n_weeks),
            five,
        ),
        ("weeks_1_3__10_edps", tuple(range(min(3, n_weeks))), all_edps),
        ("weeks_1_12__10_edps", tuple(range(min(12, n_weeks))), all_edps),
        ("all_weeks__10_edps", tuple(range(n_weeks)), all_edps),
    )


def provider_training_specs(n_edps: int, n_weeks: int):
    wanted = {
        "weeks_1_3__2_edps",
        "all_weeks__2_edps",
        "weeks_7_13__5_edps",
        "all_weeks__5_edps",
        "weeks_1_12__10_edps",
        "all_weeks__10_edps",
    }
    return tuple(spec for spec in provider_report_specs(n_edps, n_weeks) if spec[0] in wanted)


def _campaign_observations(world, campaigns, specs):
    return [
        measure_report(world, campaign, weeks, edps)
        for campaign in campaigns
        for _, weeks, edps in specs
    ]


def _make_panel_campaigns(world, per_scenario: int, seed_offset: int, prefix: str):
    campaigns = []
    for scenario_index, scenario in enumerate(META_CAMPAIGN_SCENARIOS):
        for replicate in range(per_scenario):
            campaigns.append(
                generate_campaign(
                    world,
                    scenario,
                    world.config.seed
                    + seed_offset
                    + scenario_index * 10_000
                    + replicate,
                    f"{prefix}_{scenario}_{replicate:02d}",
                )
            )
    return campaigns


def _fit_comparators(world):
    config = world.config
    campaigns = [
        generate_campaign(
            world,
            "representative",
            config.seed + 10_000 + index,
            f"legacy_calibration_{index:03d}",
        )
        for index in range(config.calibration_train_campaigns)
    ]
    observations = [
        measure_report(
            world,
            campaign,
            weeks,
            tuple(range(config.n_edps)),
        )
        for campaign in campaigns
        for weeks in calibration_checkpoints(config.n_weeks)
    ]
    data = calibration_dataset(observations, config.minimum_calibration_intersection)
    return (
        DirectPairLogModel.fit(data, config.n_edps, config.ridge_penalty),
        LatentMixtureModel.fit(data, config.n_edps, config.seed + 902),
    )


def _mean_by(rows, category: str, key: str):
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["category"] == category:
            grouped[str(row[key])].append(float(row["value"]))
    return {name: summarize(values) for name, values in sorted(grouped.items())}


def _method_summary(rows, methods):
    result = []
    for method in methods:
        total = [
            float(row["value"])
            for row in rows
            if row["category"] == "total_error" and row["method"] == method["name"]
        ]
        demo_reach = [
            float(row["value"])
            for row in rows
            if row["category"] == "demographic_reach_error"
            and row["method"] == method["name"]
        ]
        demo_distribution = [
            float(row["value"])
            for row in rows
            if row["category"] == "demographic_distribution_error"
            and row["method"] == method["name"]
        ]
        result.append(
            {
                **method,
                "total_error": summarize(total),
                "demographic_reach_error": summarize(demo_reach),
                "demographic_distribution_error": summarize(demo_distribution),
                "total_error_by_edp_count": {
                    str(count): summarize(
                        [
                            float(row["value"])
                            for row in rows
                            if row["category"] == "total_error"
                            and row["method"] == method["name"]
                            and int(row["edp_count"]) == count
                        ]
                    )
                    for count in (2, 5, 10)
                },
            }
        )
    return result


def _evaluate_campaigns(
    world,
    campaigns,
    specs,
    direct_pair,
    mixture,
    provider_email,
    provider_context,
    proportional,
    fixed_allocator,
    contextual_allocator,
    split: str,
):
    rows = []
    consistency_values: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    method_definitions = (
        {
            "name": "existing_vid",
            "label": "Existing VID",
            "category": "baseline",
            "description": "Existing population-rate total and existing VID demographics.",
        },
        {
            "name": "direct_pair_proportional",
            "label": "Direct pair inference + proportional demos",
            "category": "measurement-layer calibration",
            "description": "Calibrate pair overlaps, infer higher orders, then scale VID demographics proportionally.",
        },
        {
            "name": "mixture_pair_proportional",
            "label": "Mixture pair inference + proportional demos",
            "category": "measurement-layer calibration",
            "description": "Use the matchability mixture for pair capture, infer higher orders, then scale demographics.",
        },
        {
            "name": "provider_email_proportional",
            "label": "Provider email-first total + proportional demos",
            "category": "provider total model",
            "description": "Panel-trained total model using email-derived VID overlap without campaign context.",
        },
        {
            "name": "provider_context_proportional",
            "label": "Provider total with context + proportional demos",
            "category": "provider total model",
            "description": "Panel-trained total model adds objective and audience-strategy inputs.",
        },
        {
            "name": "provider_context_fixed_demo",
            "label": "Provider total + fixed panel demo adjustment",
            "category": "provider total and demographic model",
            "description": "Uses one panel-learned correction per demographic cell.",
        },
        {
            "name": "provider_context_learned_demo",
            "label": "Provider total + contextual panel demo adjustment",
            "category": "provider total and demographic model",
            "description": "Adjusts VID demographic shares using panel and campaign context.",
        },
        {
            "name": "oracle_total_proportional",
            "label": "Oracle total + proportional demos",
            "category": "diagnostic oracle",
            "description": "Uses true synthetic total to isolate proportional allocation error.",
        },
        {
            "name": "oracle_total_learned_demo",
            "label": "Oracle total + contextual panel demos",
            "category": "diagnostic oracle",
            "description": "Uses true synthetic total to isolate the remaining demographic-allocation error.",
        },
    )

    for campaign in campaigns:
        for report_label, weeks, edps in specs:
            observation = measure_report(world, campaign, weeks, edps)
            truth_total = float(observation.truth_unions[-1])
            started = perf_counter()
            direct_total = calibrate_report_pairwise_maximum_entropy(
                observation,
                direct_pair,
                pair_ridge=1e-6,
                evidence_half_saturation=0.1,
                name="direct_pair",
            ).full_union
            direct_runtime = perf_counter() - started
            started = perf_counter()
            mixture_total = calibrate_report_pairwise_maximum_entropy(
                observation,
                mixture,
                pair_ridge=1e-6,
                evidence_half_saturation=0.1,
                name="mixture_pair",
            ).full_union
            mixture_runtime = perf_counter() - started
            started = perf_counter()
            provider_email_total = provider_email.predict(observation)
            email_runtime = perf_counter() - started
            started = perf_counter()
            provider_context_total = provider_context.predict(observation)
            context_runtime = perf_counter() - started

            totals = {
                "existing_vid": float(observation.baseline_unions[-1]),
                "direct_pair_proportional": direct_total,
                "mixture_pair_proportional": mixture_total,
                "provider_email_proportional": provider_email_total,
                "provider_context_proportional": provider_context_total,
                "provider_context_fixed_demo": provider_context_total,
                "provider_context_learned_demo": provider_context_total,
                "oracle_total_proportional": truth_total,
                "oracle_total_learned_demo": truth_total,
            }
            runtimes = {
                "existing_vid": 0.0,
                "direct_pair_proportional": direct_runtime,
                "mixture_pair_proportional": mixture_runtime,
                "provider_email_proportional": email_runtime,
                "provider_context_proportional": context_runtime,
                "provider_context_fixed_demo": context_runtime,
                "provider_context_learned_demo": context_runtime,
                "oracle_total_proportional": 0.0,
                "oracle_total_learned_demo": 0.0,
            }
            demographic_estimates = {
                "existing_vid": observation.baseline_demographic_union,
                "direct_pair_proportional": proportional.allocate(direct_total, observation),
                "mixture_pair_proportional": proportional.allocate(mixture_total, observation),
                "provider_email_proportional": proportional.allocate(
                    provider_email_total,
                    observation,
                ),
                "provider_context_proportional": proportional.allocate(
                    provider_context_total,
                    observation,
                ),
                "provider_context_fixed_demo": fixed_allocator.allocate(
                    provider_context_total,
                    observation,
                ),
                "provider_context_learned_demo": contextual_allocator.allocate(
                    provider_context_total,
                    observation,
                ),
                "oracle_total_proportional": proportional.allocate(truth_total, observation),
                "oracle_total_learned_demo": contextual_allocator.allocate(
                    truth_total,
                    observation,
                ),
            }

            for method in method_definitions:
                name = method["name"]
                estimate = totals[name]
                demographic = demographic_estimates[name]
                common = {
                    "split": split,
                    "scenario": campaign.scenario,
                    "scenario_label": SCENARIO_LABELS[campaign.scenario],
                    "campaign": campaign.campaign_id,
                    "report": report_label,
                    "week_count": len(weeks),
                    "edp_count": len(edps),
                    "method": name,
                    "method_label": method["label"],
                    "truth_total": truth_total,
                    "estimate_total": estimate,
                    "runtime_seconds": runtimes[name],
                }
                rows.append(
                    {
                        **common,
                        "category": "total_error",
                        "value": relative_error(estimate, truth_total),
                    }
                )
                rows.append(
                    {
                        **common,
                        "category": "demographic_reach_error",
                        "value": demographic_reach_error(
                            demographic,
                            observation.truth_demographic_union,
                        ),
                    }
                )
                rows.append(
                    {
                        **common,
                        "category": "demographic_distribution_error",
                        "value": demographic_distribution_error(
                            demographic,
                            observation.truth_demographic_union,
                        ),
                    }
                )
                rows.append(
                    {
                        **common,
                        "category": "demographic_sum_error",
                        "value": abs(float(np.sum(demographic)) - estimate)
                        / max(estimate, 1.0),
                    }
                )
                rows.append(
                    {
                        **common,
                        "category": "demographic_population_violation",
                        "value": float(
                            np.max(
                                np.maximum(
                                    demographic - observation.demographic_population,
                                    0.0,
                                )
                            )
                            / max(truth_total, 1.0)
                        ),
                    }
                )
                consistency_values[(campaign.campaign_id, name)][report_label] = estimate

    nested_pairs = (
        ("weeks_1_3__10_edps", "weeks_1_12__10_edps"),
        ("weeks_1_12__10_edps", "all_weeks__10_edps"),
        ("all_weeks__2_edps", "all_weeks__5_edps"),
        ("all_weeks__5_edps", "all_weeks__10_edps"),
    )
    consistency = defaultdict(lambda: {"checks": 0, "violations": 0, "maximum": 0.0})
    for (_, method), values in consistency_values.items():
        for smaller, larger in nested_pairs:
            if smaller not in values or larger not in values:
                continue
            violation = max(values[smaller] - values[larger], 0.0)
            consistency[method]["checks"] += 1
            consistency[method]["violations"] += int(violation > 1e-6)
            consistency[method]["maximum"] = max(
                consistency[method]["maximum"],
                violation,
            )
    return rows, list(method_definitions), dict(consistency)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(rows, methods, category, path, title, axis_label, method_names):
    scenarios = list(META_CAMPAIGN_SCENARIOS)
    production = [method for method in methods if method["name"] in method_names]
    matrix = np.zeros((len(production), len(scenarios)), dtype=float)
    for method_index, method in enumerate(production):
        for scenario_index, scenario in enumerate(scenarios):
            values = [
                float(row["value"])
                for row in rows
                if row["split"] == "evaluation"
                and row["category"] == category
                and row["method"] == method["name"]
                and row["scenario"] == scenario
            ]
            matrix[method_index, scenario_index] = float(np.mean(values))
    width = 0.11
    positions = np.arange(len(scenarios))
    figure, axis = plt.subplots(figsize=(17, 7))
    for method_index, method in enumerate(production):
        offset = (method_index - (len(production) - 1) / 2.0) * width
        axis.bar(positions + offset, 100.0 * matrix[method_index], width, label=method["label"])
    axis.set_xticks(positions, [SCENARIO_LABELS[item] for item in scenarios], rotation=38, ha="right")
    axis.set_ylabel(axis_label)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def run_provider_benchmark(
    output_dir: Path,
    profile: str = "quick",
    panel_train_per_scenario: int | None = None,
    panel_holdout_per_scenario: int | None = None,
    evaluation_per_scenario: int | None = None,
):
    config = SimulationConfig.for_profile(profile)
    if panel_train_per_scenario is None:
        panel_train_per_scenario = 2 if profile == "quick" else 3
    if panel_holdout_per_scenario is None:
        panel_holdout_per_scenario = 1 if profile == "quick" else 2
    if evaluation_per_scenario is None:
        evaluation_per_scenario = 2 if profile == "quick" else 3
    output_dir.mkdir(parents=True, exist_ok=True)
    world = make_world(config)
    training_specs = provider_training_specs(config.n_edps, config.n_weeks)
    evaluation_specs = provider_report_specs(config.n_edps, config.n_weeks)

    panel_train_campaigns = _make_panel_campaigns(
        world,
        panel_train_per_scenario,
        5_000_000,
        "panel_train",
    )
    panel_holdout_campaigns = _make_panel_campaigns(
        world,
        panel_holdout_per_scenario,
        6_000_000,
        "panel_holdout",
    )
    evaluation_campaigns = _make_panel_campaigns(
        world,
        evaluation_per_scenario,
        7_000_000,
        "evaluation",
    )
    panel_train_observations = _campaign_observations(
        world,
        panel_train_campaigns,
        training_specs,
    )

    provider_email = PanelTotalReachModel.fit(
        panel_train_observations,
        config.n_edps,
        include_context=False,
    )
    provider_context = PanelTotalReachModel.fit(
        panel_train_observations,
        config.n_edps,
        include_context=True,
    )
    proportional = ProportionalDemographicAllocator()
    fixed_allocator = FixedDemographicAllocator.fit(panel_train_observations)
    contextual_allocator = ContextualDemographicAllocator.fit(
        panel_train_observations,
        config.n_edps,
    )
    direct_pair, mixture = _fit_comparators(world)

    holdout_rows, methods, holdout_consistency = _evaluate_campaigns(
        world,
        panel_holdout_campaigns,
        evaluation_specs,
        direct_pair,
        mixture,
        provider_email,
        provider_context,
        proportional,
        fixed_allocator,
        contextual_allocator,
        "panel_holdout",
    )
    evaluation_rows, _, evaluation_consistency = _evaluate_campaigns(
        world,
        evaluation_campaigns,
        evaluation_specs,
        direct_pair,
        mixture,
        provider_email,
        provider_context,
        proportional,
        fixed_allocator,
        contextual_allocator,
        "evaluation",
    )
    rows = holdout_rows + evaluation_rows

    production_methods = [method for method in methods if method["category"] != "diagnostic oracle"]
    evaluation_summary = _method_summary(
        [row for row in rows if row["split"] == "evaluation"],
        methods,
    )
    holdout_summary = _method_summary(
        [row for row in rows if row["split"] == "panel_holdout"],
        methods,
    )
    scenario_summary = {}
    for scenario in META_CAMPAIGN_SCENARIOS:
        scenario_summary[scenario] = {}
        for method in methods:
            scenario_summary[scenario][method["name"]] = {
                category: summarize(
                    [
                        float(row["value"])
                        for row in evaluation_rows
                        if row["scenario"] == scenario
                        and row["method"] == method["name"]
                        and row["category"] == category
                    ]
                )
                for category in (
                    "total_error",
                    "demographic_reach_error",
                    "demographic_distribution_error",
                )
            }

    total_chart = output_dir / "provider_total_error_by_scenario.png"
    demographic_chart = output_dir / "provider_demographic_error_by_scenario.png"
    _plot_metric(
        rows,
        methods,
        "total_error",
        total_chart,
        "Total union-reach error by campaign scenario",
        "Mean absolute relative error (%)",
        (
            "existing_vid",
            "direct_pair_proportional",
            "mixture_pair_proportional",
            "provider_email_proportional",
            "provider_context_proportional",
        ),
    )
    _plot_metric(
        rows,
        methods,
        "demographic_distribution_error",
        demographic_chart,
        "Demographic distribution error by campaign scenario",
        "Audience share assigned to the wrong demographic cells (%)",
        (
            "existing_vid",
            "provider_context_proportional",
            "provider_context_fixed_demo",
            "provider_context_learned_demo",
        ),
    )

    summary = {
        "profile": profile,
        "config": config.__dict__,
        "panel_train_campaigns": len(panel_train_campaigns),
        "panel_holdout_campaigns": len(panel_holdout_campaigns),
        "evaluation_campaigns": len(evaluation_campaigns),
        "training_observations": len(panel_train_observations),
        "demographic_cells": len(world.demographic_labels),
        "demographic_labels": world.demographic_labels,
        "methods": evaluation_summary,
        "holdout_methods": holdout_summary,
        "scenario_summary": scenario_summary,
        "consistency": {
            "panel_holdout": holdout_consistency,
            "evaluation": evaluation_consistency,
        },
        "provider_models": {
            "email_only_parameter_count": provider_email.parameter_count,
            "context_parameter_count": provider_context.parameter_count,
            "contextual_demographic_parameter_count": contextual_allocator.parameter_count,
        },
        "production_method_count": len(production_methods),
    }
    _write_csv(output_dir / "provider_metrics.csv", rows)
    (output_dir / "provider_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare provider-owned total-reach and demographic calibration"
    )
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/provider_model"),
    )
    parser.add_argument("--panel-train-per-scenario", type=int)
    parser.add_argument("--panel-holdout-per-scenario", type=int)
    parser.add_argument("--evaluation-per-scenario", type=int)
    arguments = parser.parse_args()
    summary = run_provider_benchmark(
        arguments.output_dir,
        arguments.profile,
        arguments.panel_train_per_scenario,
        arguments.panel_holdout_per_scenario,
        arguments.evaluation_per_scenario,
    )
    print(
        json.dumps(
            {
                "methods": len(summary["methods"]),
                "demographic_cells": summary["demographic_cells"],
                "output_dir": str(arguments.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
