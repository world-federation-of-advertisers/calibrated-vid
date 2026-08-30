from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .daily_labeling import (
    LabelingResult,
    _mix64,
    _report_union,
    _truth_union,
    generate_temporal_stress_campaign,
    label_oracle_online_venn,
    report_specs,
)
from .population import Campaign, generate_campaign, make_world
from .sets import exact_cells_from_membership


SCENARIOS = (
    "broad_awareness_control",
    "traffic_optimization",
    "video_engagement_retargeting",
    "lead_generation",
    "sales_prospecting",
    "website_retargeting",
    "crm_customer_list",
    "catalog_retargeting",
    "lookalike_prospecting",
    "advantage_audience_expansion",
    "app_activity_retargeting",
    "unrelated_niche_control",
    "mixed_funnel_portfolio",
    "linkage_shift_-1.0",
    "linkage_shift_0.0",
    "linkage_shift_1.0",
)


def _label_exact_cells(
    labels: np.ndarray,
    weeks: tuple[int, ...],
    edps: tuple[int, ...],
) -> np.ndarray:
    users = np.arange(labels.shape[2])
    membership: dict[int, int] = {}
    for local, edp in enumerate(edps):
        values = labels[np.ix_((edp,), weeks, users)].reshape(-1)
        for value in np.unique(values[values >= 0]).tolist():
            membership[int(value)] = membership.get(int(value), 0) | (1 << local)
    return np.bincount(
        np.fromiter(membership.values(), dtype=np.int64),
        minlength=1 << len(edps),
    ).astype(int)


def _truth_exact_cells(
    campaign: Campaign,
    weeks: tuple[int, ...],
    edps: tuple[int, ...],
) -> np.ndarray:
    users = np.arange(campaign.events.shape[2])
    membership = np.any(campaign.events[np.ix_(edps, weeks, users)], axis=1)
    return exact_cells_from_membership(membership, 1.0).astype(int)


def label_daily_full_venn(campaign: Campaign) -> LabelingResult:
    """Match each day's complete EDP Venn table, without cross-day identity data."""
    n_edps, n_weeks, n_users = campaign.events.shape
    labels = np.full(campaign.events.shape, -1, dtype=np.int64)
    next_vid = 0
    all_edps = tuple(range(n_edps))
    for day in range(n_weeks):
        queues: dict[int, list[int]] = {}
        for edp in all_edps:
            users = np.flatnonzero(campaign.events[edp, day])
            queues[edp] = sorted(
                users.tolist(),
                key=lambda user: int(_mix64((edp + 1) * n_users + user + 1, 0xDA117)[()]),
            )
        daily_membership = campaign.events[:, day]
        target = exact_cells_from_membership(daily_membership, 1.0).astype(int)
        for mask in sorted(range(1, 1 << n_edps), key=lambda value: (-value.bit_count(), value)):
            count = int(target[mask])
            if not count:
                continue
            vids = np.arange(next_vid, next_vid + count, dtype=np.int64)
            next_vid += count
            for edp in range(n_edps):
                if not mask & (1 << edp):
                    continue
                selected = queues[edp][:count]
                del queues[edp][:count]
                labels[edp, day, np.asarray(selected, dtype=int)] = vids
        if any(queues.values()):
            raise RuntimeError("daily Venn allocation did not consume every EDP event")
    return LabelingResult(
        "daily_full_venn",
        labels,
        np.zeros(n_weeks, dtype=float),
        0,
        "Every daily Venn cell is exact; cross-day identity is deliberately unavailable.",
        supported_edps=n_edps,
        state_entries=next_vid,
        pool_count=(1 << n_edps) - 1,
        requires_ordered_days=False,
    )


def indistinguishable_window_counterexample() -> dict:
    """Two worlds with identical daily and cumulative counts but different intervals."""
    first = ({"a"}, {"b"}, {"a"})
    second = ({"a"}, {"b"}, {"b"})

    def daily(world):
        return [len(value) for value in world]

    def cumulative(world):
        seen: set[str] = set()
        output = []
        for value in world:
            seen |= value
            output.append(len(seen))
        return output

    return {
        "world_1": [[*sorted(value)] for value in first],
        "world_2": [[*sorted(value)] for value in second],
        "daily_reach_both": daily(first),
        "cumulative_reach_both": cumulative(first),
        "world_1_weeks_2_3_union": len(first[1] | first[2]),
        "world_2_weeks_2_3_union": len(second[1] | second[2]),
        "daily_equal": daily(first) == daily(second),
        "cumulative_equal": cumulative(first) == cumulative(second),
    }


def time_atom_audit(campaign: Campaign, n_edps: int = 3, n_weeks: int = 4) -> dict:
    """Show that complete EDP-by-time activity atoms answer every report exactly."""
    events = campaign.events[:n_edps, :n_weeks]
    signatures = np.zeros(events.shape[2], dtype=object)
    for edp in range(n_edps):
        for week in range(n_weeks):
            signatures += events[edp, week].astype(object) * (1 << (edp * n_weeks + week))
    values, counts = np.unique(signatures.astype(int), return_counts=True)
    atom_counts = {int(value): int(count) for value, count in zip(values, counts) if int(value)}
    anonymous_event_sets: list[set[int]] = [set() for _ in range(n_edps * n_weeks)]
    next_vid = 0
    for signature, count in sorted(atom_counts.items()):
        vids = set(range(next_vid, next_vid + count))
        next_vid += count
        for bit in range(n_edps * n_weeks):
            if signature & (1 << bit):
                anonymous_event_sets[bit].update(vids)

    checks = 0
    max_difference = 0
    for edp_mask in range(1, 1 << n_edps):
        selected_edps = tuple(edp for edp in range(n_edps) if edp_mask & (1 << edp))
        for week_mask in range(1, 1 << n_weeks):
            selected_weeks = tuple(week for week in range(n_weeks) if week_mask & (1 << week))
            truth = _truth_union(campaign, selected_weeks, selected_edps)
            selected_sets = [
                anonymous_event_sets[edp * n_weeks + week]
                for edp in selected_edps
                for week in selected_weeks
            ]
            estimate = len(set().union(*selected_sets))
            max_difference = max(max_difference, abs(estimate - truth))
            checks += 1
    return {
        "edps": n_edps,
        "weeks": n_weeks,
        "possible_nonempty_activity_atoms": (1 << (n_edps * n_weeks)) - 1,
        "observed_nonempty_activity_atoms": len(atom_counts),
        "report_queries_checked": checks,
        "max_report_difference": max_difference,
    }


def _campaigns(world, seed: int) -> list[Campaign]:
    campaigns = [
        generate_campaign(world, scenario, seed + index * 101, f"venn_{scenario}")
        for index, scenario in enumerate(SCENARIOS)
    ]
    campaigns.extend(
        generate_temporal_stress_campaign(
            world,
            scenario,
            seed + 50_000 + index,
            f"venn_{scenario}",
        )
        for index, scenario in enumerate(
            ("staggered_retargeting", "synchronized_retargeting", "shared_seed_then_expansion")
        )
    )
    return campaigns


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _proof_report_specs(n_edps: int, n_weeks: int):
    base = list(report_specs(n_edps, n_weeks))
    all_edps = tuple(range(n_edps))
    base.extend(
        [
            ("weeks_5_12__10_edps", tuple(range(4, min(12, n_weeks))), all_edps),
            ("weeks_7_13__10_edps", tuple(range(6, n_weeks)), all_edps),
            (
                "noncontiguous__10_edps",
                tuple(index for index in (0, 2, 4, 7, 10, 12) if index < n_weeks),
                all_edps,
            ),
            (
                "noncontiguous__2_edps",
                tuple(index for index in (0, 2, 4, 7, 10, 12) if index < n_weeks),
                (0, 1),
            ),
        ]
    )
    return tuple(base)


def _plot(rows: list[dict], output: Path) -> None:
    labels = {
        "daily_full_venn": "Full daily Venn only",
        "cumulative_full_venn_recent": "Full cumulative Venn / prefer recent",
        "cumulative_full_venn_oldest": "Full cumulative Venn / prefer old",
    }
    report_types = ("prefix", "interval", "noncontiguous")
    x = np.arange(len(report_types))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index, method in enumerate(labels):
        means = [
            100.0
            * np.mean(
                [
                    row["union_relative_error"]
                    for row in rows
                    if row["method"] == method and row["report_type"] == report_type
                ]
            )
            for report_type in report_types
        ]
        ax.bar(x + (index - 1) * width, means, width, label=labels[method])
    ax.set_xticks(x, [value.title() for value in report_types])
    ax.set_ylabel("Mean union-reach error (%)")
    ax.set_title("What full daily or cumulative Venn information guarantees")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_full_venn_proof(output_dir: Path, profile: str = "full") -> dict:
    n_users = 18_000 if profile == "full" else 4_000
    config = SimulationConfig(
        n_users=n_users,
        population_size=180_000_000,
        n_edps=10,
        n_weeks=13,
        seed=20260830,
    )
    world = make_world(config)
    campaigns = _campaigns(world, config.seed + 1_000)
    rows: list[dict] = []
    prefix_cell_differences: dict[str, int] = {}
    daily_cell_difference = 0

    for campaign in campaigns:
        methods = [
            label_daily_full_venn(campaign),
            label_oracle_online_venn(world, campaign, edp_count=10, prefer_recent_slots=True),
            label_oracle_online_venn(world, campaign, edp_count=10, prefer_recent_slots=False),
        ]
        methods[1] = LabelingResult(
            "cumulative_full_venn_recent",
            methods[1].labels,
            methods[1].day_dials,
            methods[1].available_day,
            methods[1].notes,
            supported_edps=10,
            state_entries=methods[1].state_entries,
            pool_count=methods[1].pool_count,
            requires_ordered_days=True,
        )
        methods[2] = LabelingResult(
            "cumulative_full_venn_oldest",
            methods[2].labels,
            methods[2].day_dials,
            methods[2].available_day,
            methods[2].notes,
            supported_edps=10,
            state_entries=methods[2].state_entries,
            pool_count=methods[2].pool_count,
            requires_ordered_days=True,
        )

        all_edps = tuple(range(config.n_edps))
        for day in range(config.n_weeks):
            weeks = tuple(range(day + 1))
            truth_cells = _truth_exact_cells(campaign, weeks, all_edps)
            for result in methods[1:]:
                estimate_cells = _label_exact_cells(result.labels, weeks, all_edps)
                difference = int(np.max(np.abs(estimate_cells[1:] - truth_cells[1:])))
                prefix_cell_differences[result.method] = max(
                    prefix_cell_differences.get(result.method, 0),
                    difference,
                )
            daily_truth = _truth_exact_cells(campaign, (day,), all_edps)
            daily_estimate = _label_exact_cells(methods[0].labels, (day,), all_edps)
            daily_cell_difference = max(
                daily_cell_difference,
                int(np.max(np.abs(daily_estimate[1:] - daily_truth[1:]))),
            )

        for result in methods:
            for report_name, weeks, edps in _proof_report_specs(
                config.n_edps,
                config.n_weeks,
            ):
                truth = _truth_union(campaign, weeks, edps)
                estimate = _report_union(result.labels, weeks, edps)
                contiguous = weeks == tuple(range(min(weeks), max(weeks) + 1))
                report_type = (
                    "prefix"
                    if weeks == tuple(range(max(weeks) + 1))
                    else "interval"
                    if contiguous
                    else "noncontiguous"
                )
                rows.append(
                    {
                        "campaign_id": campaign.campaign_id,
                        "scenario": campaign.scenario,
                        "method": result.method,
                        "report": report_name,
                        "report_type": report_type,
                        "edp_count": len(edps),
                        "week_count": len(weeks),
                        "truth_union": truth,
                        "estimated_union": estimate,
                        "union_relative_error": abs(estimate - truth) / max(truth, 1),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "full_venn_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    method_summary = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        method_summary[method] = {
            "all_reports": _summary([row["union_relative_error"] for row in selected]),
            "by_report_type": {
                report_type: _summary(
                    [
                        row["union_relative_error"]
                        for row in selected
                        if row["report_type"] == report_type
                    ]
                )
                for report_type in ("prefix", "interval", "noncontiguous")
            },
            "by_edp_count": {
                str(edp_count): _summary(
                    [
                        row["union_relative_error"]
                        for row in selected
                        if row["edp_count"] == edp_count
                    ]
                )
                for edp_count in (2, 5, 10)
            },
        }

    summary = {
        "configuration": {
            "profile": profile,
            "n_users": n_users,
            "n_edps": config.n_edps,
            "n_weeks": config.n_weeks,
            "campaign_count": len(campaigns),
            "cumulative_venn_cell_count": (1 << config.n_edps) - 1,
        },
        "methods": method_summary,
        "exact_cell_audits": {
            "daily_full_venn_max_cell_difference": daily_cell_difference,
            **{
                f"{method}_max_prefix_cell_difference": value
                for method, value in prefix_cell_differences.items()
            },
        },
        "daily_plus_cumulative_counterexample": indistinguishable_window_counterexample(),
        "time_atom_audit": time_atom_audit(campaigns[0]),
        "ten_edp_thirteen_week_possible_activity_atoms": (1 << (10 * 13)) - 1,
    }
    (output_dir / "full_venn_summary.json").write_text(json.dumps(summary, indent=2))
    _plot(rows, output_dir / "full_venn_report_error.png")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run_full_venn_proof(arguments.output_dir, arguments.profile)
