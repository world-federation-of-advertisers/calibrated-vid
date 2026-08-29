from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .calibrated_venn_labeling import _evaluation_campaigns
from .config import SimulationConfig
from .population import make_world


def two_day_anchor_counterexample() -> dict[str, object]:
    """Conditional stress case when an EDP can change its acting key over time.

    On day one, a proprietary identifier at A is deliberately overlapped with
    an email identifier at B.  If that same email later arrives at A, its
    stable cross-EDP VID is already occupied at A by a different identifier.
    Keeping the email anchor collapses A's two identifiers to one VID; moving
    the email preserves A's reach but breaks the anchor.
    """
    return {
        "assumption": (
            "The same EDP can first label a person with a proprietary fallback and later "
            "present a shared email-derived Reference ID for that person, or a synthetic "
            "fallback assignment may occupy a Reference-ID VID at that EDP. A stable "
            "pre-ranked Reference-ID lane avoids this premise."
        ),
        "day_1": {
            "edp_a": ["proprietary:p"],
            "edp_b": ["email:e"],
            "required_union": 1,
            "forced_assignment": "VID(p@A) = VID(e@B)",
        },
        "day_2": {
            "edp_a_adds": ["email:e"],
            "required_edp_a_reach": 2,
        },
        "conditional_conflict": (
            "Stable email identity requires VID(e@A) = VID(e@B), but the day-1 "
            "overlap already made that VID equal to VID(p@A). EDP A would then "
            "have one VID for two distinct identifiers. Assigning a second VID "
            "instead preserves A's reach but breaks the shared-email anchor."
        ),
        "ways_out": (
            "reserve capacity for future email arrivals",
            "know the relevant identity roster before finalizing labels",
            "permit a controlled anchor miss or local collision",
            "permit later relabeling or restatement",
        ),
    }


def run_online_identity_constraints(output_dir: Path) -> dict[str, object]:
    config = SimulationConfig(
        n_users=12_000,
        population_size=180_000_000,
        n_edps=10,
        n_weeks=13,
        seed=20260831,
    )
    world = make_world(config)
    campaigns = _evaluation_campaigns(world)
    rows: list[dict[str, object]] = []

    for campaign in campaigns:
        reached = np.any(campaign.events, axis=1)
        email = reached & world.email_linkable
        proprietary = reached & ~world.email_linkable
        globally_used_email = np.any(email, axis=0)
        strict_global_capacity = config.n_users - int(globally_used_email.sum())
        proprietary_counts = proprietary.sum(axis=1)
        email_counts = email.sum(axis=1)

        pair_cross_mode: list[float] = []
        for left in range(config.n_edps):
            for right in range(left + 1, config.n_edps):
                truth_overlap = int(np.sum(reached[left] & reached[right]))
                if truth_overlap == 0:
                    continue
                cross_mode = int(
                    np.sum(
                        (email[left] & proprietary[right])
                        | (proprietary[left] & email[right])
                    )
                )
                pair_cross_mode.append(cross_mode / truth_overlap)

        for edp in range(config.n_edps):
            oracle_capacity = config.n_users - int(email_counts[edp])
            rows.append(
                {
                    "campaign": campaign.campaign_id,
                    "scenario": campaign.scenario,
                    "edp": edp,
                    "global_email_reservation_capacity": strict_global_capacity,
                    "proprietary_identifiers": int(proprietary_counts[edp]),
                    "global_reservation_feasible": bool(
                        proprietary_counts[edp] <= strict_global_capacity
                    ),
                    "future_aware_reservation_capacity": oracle_capacity,
                    "future_aware_reservation_feasible": bool(
                        proprietary_counts[edp] <= oracle_capacity
                    ),
                    "mean_cross_mode_share_of_true_pair_overlap": float(
                        np.mean(pair_cross_mode) if pair_cross_mode else 0.0
                    ),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "online_identity_constraints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    global_feasible = [bool(row["global_reservation_feasible"]) for row in rows]
    future_feasible = [bool(row["future_aware_reservation_feasible"]) for row in rows]
    cross_mode = [
        float(row["mean_cross_mode_share_of_true_pair_overlap"])
        for row in rows[:: config.n_edps]
    ]
    capacity_ratios = [
        float(row["proprietary_identifiers"])
        / max(float(row["global_email_reservation_capacity"]), 1.0)
        for row in rows
    ]
    summary: dict[str, object] = {
        "configuration": {
            "n_users": config.n_users,
            "n_edps": config.n_edps,
            "n_weeks": config.n_weeks,
            "campaigns": len(campaigns),
        },
        "two_day_counterexample": two_day_anchor_counterexample(),
        "reserve_every_email_vid_at_every_edp": {
            "feasible_edp_campaign_fraction": float(np.mean(global_feasible)),
            "campaigns_with_all_edps_feasible": int(
                sum(
                    all(global_feasible[start : start + config.n_edps])
                    for start in range(0, len(global_feasible), config.n_edps)
                )
            ),
            "maximum_proprietary_demand_to_capacity": float(max(capacity_ratios)),
        },
        "future_aware_per_edp_reservation": {
            "feasible_edp_campaign_fraction": float(np.mean(future_feasible)),
            "campaigns_with_all_edps_feasible": int(
                sum(
                    all(future_feasible[start : start + config.n_edps])
                    for start in range(0, len(future_feasible), config.n_edps)
                )
            ),
        },
        "strict_email_and_proprietary_namespaces": {
            "mean_true_pair_overlap_that_crosses_identifier_modes": float(np.mean(cross_mode)),
            "p90_true_pair_overlap_that_crosses_identifier_modes": float(
                np.quantile(cross_mode, 0.9)
            ),
            "interpretation": (
                "This share cannot be represented as an email-to-proprietary match if the two "
                "VID namespaces never overlap."
            ),
        },
    }
    (output_dir / "online_identity_constraints.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run_online_identity_constraints(arguments.output_dir)
