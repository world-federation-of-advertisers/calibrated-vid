from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


EDPS = ("EDP_A", "EDP_B")


@dataclass(frozen=True)
class Impression:
    day: int
    edp: str
    event_id: str
    account_id: str
    email: str | None = None
    reference_id: int | None = None


@dataclass(frozen=True)
class Assignment:
    day: int
    edp: str
    event_id: str
    account_id: str
    reference_id: int | None
    local_rank: int
    symbolic_vid: str
    assignment_reason: str
    is_new_account: bool
    warning: str | None = None


@dataclass(frozen=True)
class DaySummary:
    day: int
    requested_overlap: int
    achieved_overlap: int
    edp_a_reach: int
    edp_b_reach: int
    union_reach: int
    direct_reference_overlap: int
    synthetic_fallback_overlap: int
    target_status: str
    late_anchor_conflicts: int


@dataclass(frozen=True)
class CoverageFeasibility:
    email_coverage_a: float
    email_coverage_b: float
    conditional_email_agreement: float
    edp_a_reach: int
    edp_b_reach: int
    target_overlap: int
    direct_reference_overlap: int
    flexible_no_email_a: int
    flexible_no_email_b: int
    maximum_reachable_overlap: int
    shortfall: int
    feasible: bool


class TwoEdpAdaptiveAllocator:
    """A small stateful example of Reference-ID-anchored VID assignment.

    The allocator is deliberately aggregate-driven. It does not assert that a
    proprietary ID at EDP_A and one at EDP_B are the same person. It may place
    them on the same synthetic VID when the calibrated overlap target requires
    more overlap than directly matching Reference IDs provide.

    Within an EDP, an account's first assigned rank is immutable. This is the
    invariant that preserves single-publisher reach and makes later reports
    reproduce prior labels without loading prior report results.
    """

    def __init__(self, pool_offset: int = 10_000_000, rank_capacity: int = 10_000):
        self.pool_offset = pool_offset
        self.rank_capacity = rank_capacity
        self._next_rank = 0
        self.account_rank: dict[tuple[str, str], int] = {}
        self.reference_rank: dict[int, int] = {}
        self.reference_edps: dict[int, set[str]] = {}
        self.rank_edps: dict[int, set[str]] = {}
        self.rank_origin: dict[int, str] = {}
        self.assignments: list[Assignment] = []
        self.day_summaries: list[DaySummary] = []

    def _new_rank(self, origin: str) -> int:
        if self._next_rank >= self.rank_capacity:
            raise RuntimeError("ranked VID capacity exhausted")
        rank = self._next_rank
        self._next_rank += 1
        self.rank_edps[rank] = set()
        self.rank_origin[rank] = origin
        return rank

    def _bind_account(self, edp: str, account_id: str, rank: int) -> None:
        existing = self.account_rank.get((edp, account_id))
        if existing is not None and existing != rank:
            raise RuntimeError("attempted to remap a frozen EDP account")
        occupants = self.rank_edps.setdefault(rank, set())
        if edp in occupants and existing is None:
            raise RuntimeError("two distinct accounts at one EDP cannot share a rank")
        self.account_rank[(edp, account_id)] = rank
        occupants.add(edp)

    def _symbolic_vid(self, rank: int) -> str:
        return f"VID({self.pool_offset}+Feistel({rank}))"

    def _overlap(self) -> int:
        return sum(1 for occupants in self.rank_edps.values() if occupants == set(EDPS))

    def _reach(self, edp: str) -> int:
        return sum(1 for occupants in self.rank_edps.values() if edp in occupants)

    def _direct_reference_overlap(self) -> int:
        return sum(
            1
            for reference_id, edps in self.reference_edps.items()
            if edps == set(EDPS)
            and self.rank_edps.get(self.reference_rank[reference_id], set()) == set(EDPS)
        )

    def _synthetic_overlap(self) -> int:
        return self._overlap() - self._direct_reference_overlap()

    def _candidate_opposite_only_ranks(self, edp: str) -> list[int]:
        opposite = EDPS[1] if edp == EDPS[0] else EDPS[0]
        candidates = [
            rank for rank, occupants in self.rank_edps.items() if occupants == {opposite}
        ]
        return sorted(
            candidates,
            key=lambda rank: (self.rank_origin.get(rank) != "direct_reference", -rank),
        )

    def assign_day(
        self,
        day: int,
        impressions: Iterable[Impression],
        requested_cumulative_overlap: int,
    ) -> list[Assignment]:
        events = sorted(
            [event for event in impressions if event.day == day],
            key=lambda event: (event.reference_id is None, event.edp, event.account_id),
        )
        if any(event.edp not in EDPS for event in events):
            raise ValueError(f"only {EDPS} are supported by this example")

        outcomes: list[Assignment] = []
        pending: dict[str, list[Impression]] = {edp: [] for edp in EDPS}
        late_anchor_conflicts = 0

        # First, reuse frozen account mappings and place directly observed
        # Reference-ID anchors. This is the identity-backed portion.
        for event in events:
            account_key = (event.edp, event.account_id)
            existing_rank = self.account_rank.get(account_key)
            warning = None
            if existing_rank is not None:
                rank = existing_rank
                reason = "reused_frozen_account_mapping"
                if event.reference_id is not None:
                    reference_rank = self.reference_rank.get(event.reference_id)
                    if reference_rank is None:
                        self.reference_rank[event.reference_id] = rank
                        self.reference_edps.setdefault(event.reference_id, set()).add(event.edp)
                        self.rank_origin[rank] = "direct_reference"
                        reason = "reused_account_and_bound_new_reference"
                    elif reference_rank != rank:
                        late_anchor_conflicts += 1
                        warning = (
                            "late_reference_conflict_preserved_existing_vid; "
                            "the overlap target must be recovered with other new assignments"
                        )
                        reason = "reused_frozen_account_mapping"
                    else:
                        self.reference_edps.setdefault(event.reference_id, set()).add(event.edp)
                outcomes.append(
                    Assignment(
                        day=day,
                        edp=event.edp,
                        event_id=event.event_id,
                        account_id=event.account_id,
                        reference_id=event.reference_id,
                        local_rank=rank,
                        symbolic_vid=self._symbolic_vid(rank),
                        assignment_reason=reason,
                        is_new_account=False,
                        warning=warning,
                    )
                )
                continue

            if event.reference_id is None:
                pending[event.edp].append(event)
                continue

            rank = self.reference_rank.get(event.reference_id)
            if rank is None:
                rank = self._new_rank("direct_reference")
                self.reference_rank[event.reference_id] = rank
                reason = "new_reference_anchor"
            else:
                reason = "matched_existing_reference_anchor"
            self.reference_edps.setdefault(event.reference_id, set()).add(event.edp)
            self._bind_account(event.edp, event.account_id, rank)
            outcomes.append(
                Assignment(
                    day=day,
                    edp=event.edp,
                    event_id=event.event_id,
                    account_id=event.account_id,
                    reference_id=event.reference_id,
                    local_rank=rank,
                    symbolic_vid=self._symbolic_vid(rank),
                    assignment_reason=reason,
                    is_new_account=True,
                )
            )

        overlap_before_fallback = self._overlap()
        feasible_target = max(requested_cumulative_overlap, overlap_before_fallback)
        remaining_needed = feasible_target - overlap_before_fallback

        # Prefer pairing two same-day proprietary-only accounts. This avoids
        # consuming an old EDP-only rank when the same overlap can be created
        # entirely from new, still-uncommitted labels.
        while pending[EDPS[0]] and pending[EDPS[1]] and remaining_needed > 0:
            left = pending[EDPS[0]].pop(0)
            right = pending[EDPS[1]].pop(0)
            rank = self._new_rank("synthetic_fallback")
            for event in (left, right):
                self._bind_account(event.edp, event.account_id, rank)
                outcomes.append(
                    Assignment(
                        day=day,
                        edp=event.edp,
                        event_id=event.event_id,
                        account_id=event.account_id,
                        reference_id=None,
                        local_rank=rank,
                        symbolic_vid=self._symbolic_vid(rank),
                        assignment_reason="new_shared_fallback_rank",
                        is_new_account=True,
                    )
                )
            remaining_needed -= 1

        # Next, place a new proprietary-only account into an already occupied
        # opposite-EDP rank when additional overlap is required. A rank backed
        # by a direct Reference ID is preferred, which is useful when email is
        # present at only one EDP. No existing label changes.
        for edp in EDPS:
            candidates = self._candidate_opposite_only_ranks(edp)
            while pending[edp] and candidates and remaining_needed > 0:
                event = pending[edp].pop(0)
                rank = candidates.pop(0)
                self.rank_origin[rank] = "synthetic_fallback"
                self._bind_account(edp, event.account_id, rank)
                remaining_needed -= 1
                outcomes.append(
                    Assignment(
                        day=day,
                        edp=edp,
                        event_id=event.event_id,
                        account_id=event.account_id,
                        reference_id=None,
                        local_rank=rank,
                        symbolic_vid=self._symbolic_vid(rank),
                        assignment_reason="filled_opposite_edp_only_rank",
                        is_new_account=True,
                    )
                )

        # All remaining accounts get EDP-exclusive ranks, preserving each
        # publisher's reach even when the desired overlap is not attainable.
        for edp in EDPS:
            for event in pending[edp]:
                rank = self._new_rank("edp_exclusive")
                self._bind_account(edp, event.account_id, rank)
                outcomes.append(
                    Assignment(
                        day=day,
                        edp=edp,
                        event_id=event.event_id,
                        account_id=event.account_id,
                        reference_id=None,
                        local_rank=rank,
                        symbolic_vid=self._symbolic_vid(rank),
                        assignment_reason="new_edp_exclusive_rank",
                        is_new_account=True,
                    )
                )

        achieved = self._overlap()
        if requested_cumulative_overlap < overlap_before_fallback:
            status = "PROJECTED_UP_TO_IMMUTABLE_LOWER_BOUND"
        elif remaining_needed > 0:
            status = "PROJECTED_DOWN_TO_REACHABLE_MAXIMUM"
        else:
            status = "EXACT"
        summary = DaySummary(
            day=day,
            requested_overlap=requested_cumulative_overlap,
            achieved_overlap=achieved,
            edp_a_reach=self._reach(EDPS[0]),
            edp_b_reach=self._reach(EDPS[1]),
            union_reach=self._reach(EDPS[0]) + self._reach(EDPS[1]) - achieved,
            direct_reference_overlap=self._direct_reference_overlap(),
            synthetic_fallback_overlap=self._synthetic_overlap(),
            target_status=status,
            late_anchor_conflicts=late_anchor_conflicts,
        )
        self.assignments.extend(sorted(outcomes, key=lambda row: (row.edp, row.account_id)))
        self.day_summaries.append(summary)
        self._validate_invariants()
        return sorted(outcomes, key=lambda row: (row.edp, row.account_id))

    def _validate_invariants(self) -> None:
        for edp in EDPS:
            ranks = [
                rank
                for (account_edp, _), rank in self.account_rank.items()
                if account_edp == edp
            ]
            if len(ranks) != len(set(ranks)):
                raise RuntimeError(f"single-publisher uniqueness violated for {edp}")
        for reference_id, rank in self.reference_rank.items():
            if rank not in self.rank_edps:
                raise RuntimeError(f"Reference ID {reference_id} points to an unknown rank")


def example_days() -> tuple[list[Impression], dict[int, int]]:
    """Returns a compact set of cases used in the design walkthrough."""
    events = [
        # Day 1: one direct email match, one synthetic fallback match, and
        # EDP-exclusive people.
        Impression(1, "EDP_A", "a-imp-001", "a-alice", "alice@example.test", 101),
        Impression(1, "EDP_A", "a-imp-002", "a-bob", "bob@example.test", 102),
        Impression(1, "EDP_A", "a-imp-003", "a-carol"),
        Impression(1, "EDP_A", "a-imp-004", "a-dan"),
        Impression(1, "EDP_B", "b-imp-001", "b-alice", "alice@example.test", 101),
        Impression(1, "EDP_B", "b-imp-002", "b-carol"),
        Impression(1, "EDP_B", "b-imp-003", "b-erin"),
        # Day 2: repeated accounts remain stable. Bob becomes a direct match.
        # Eve is an asymmetric-coverage case: only EDP_B has email, while the
        # aggregate allocator can still place EDP_A's proprietary account on
        # the same synthetic VID.
        Impression(2, "EDP_A", "a-imp-005", "a-alice", "alice@example.test", 101),
        Impression(2, "EDP_A", "a-imp-006", "a-carol"),
        Impression(2, "EDP_A", "a-imp-007", "a-eve"),
        Impression(2, "EDP_B", "b-imp-004", "b-alice", "alice@example.test", 101),
        Impression(2, "EDP_B", "b-imp-005", "b-bob", "bob@example.test", 102),
        Impression(2, "EDP_B", "b-imp-006", "b-eve", "eve@example.test", 303),
        # Day 3: the target asks for less overlap than the already frozen
        # labels contain. The allocator keeps old labels and flags projection.
        # EDP_A now observes Eve's email; because its proprietary rank was
        # already paired with EDP_B's Reference-ID rank, the anchor agrees.
        Impression(3, "EDP_A", "a-imp-008", "a-eve", "eve@example.test", 303),
        Impression(3, "EDP_A", "a-imp-009", "a-frank"),
        Impression(3, "EDP_B", "b-imp-007", "b-eve", "eve@example.test", 303),
        Impression(3, "EDP_B", "b-imp-008", "b-grace"),
        # Day 4 creates a deliberately adverse late-anchor case. Heidi's
        # publisher-local identities are initially placed on different VIDs.
        Impression(4, "EDP_A", "a-imp-010", "a-heidi"),
        Impression(4, "EDP_B", "b-imp-009", "b-heidi", "heidi@example.test", 404),
        # Day 5 reveals the same email at EDP_A. Remapping would corrupt old
        # single-publisher reports, so the original VID wins and the conflict
        # is surfaced for aggregate compensation/review.
        Impression(5, "EDP_A", "a-imp-011", "a-heidi", "heidi@example.test", 404),
        Impression(5, "EDP_B", "b-imp-010", "b-ivan"),
    ]
    targets = {1: 2, 2: 4, 3: 3, 4: 4, 5: 5}
    return events, targets


def coverage_feasibility(
    email_coverage_a: float,
    email_coverage_b: float,
    conditional_email_agreement: float,
    edp_a_reach: int = 600_000,
    edp_b_reach: int = 500_000,
    target_overlap: int = 300_000,
) -> CoverageFeasibility:
    """Computes the strict fixed-email-lane feasibility bound for two EDPs.

    A matched Reference ID is already shared. An unmatched email-backed rank is
    fixed and cannot be merged later under this strict design. Therefore every
    additional synthetic overlap needs at least one no-email account, which is
    the flexible residual supply.
    """
    direct = int(
        round(
            target_overlap
            * email_coverage_a
            * email_coverage_b
            * conditional_email_agreement
        )
    )
    flexible_a = int(round(edp_a_reach * (1.0 - email_coverage_a)))
    flexible_b = int(round(edp_b_reach * (1.0 - email_coverage_b)))
    remaining_a = edp_a_reach - direct
    remaining_b = edp_b_reach - direct
    additional = min(remaining_a, remaining_b, flexible_a + flexible_b)
    maximum = direct + additional
    shortfall = max(target_overlap - maximum, 0)
    return CoverageFeasibility(
        email_coverage_a=email_coverage_a,
        email_coverage_b=email_coverage_b,
        conditional_email_agreement=conditional_email_agreement,
        edp_a_reach=edp_a_reach,
        edp_b_reach=edp_b_reach,
        target_overlap=target_overlap,
        direct_reference_overlap=direct,
        flexible_no_email_a=flexible_a,
        flexible_no_email_b=flexible_b,
        maximum_reachable_overlap=maximum,
        shortfall=shortfall,
        feasible=shortfall == 0,
    )


def coverage_sweep() -> list[CoverageFeasibility]:
    coverage_pairs = (
        (0.10, 0.10),
        (0.10, 0.90),
        (0.50, 0.50),
        (0.90, 0.10),
        (0.90, 0.90),
        (0.95, 0.95),
    )
    return [coverage_feasibility(a, b, 0.60) for a, b in coverage_pairs]


def model_textproto() -> str:
    """A valid CompiledNode example for current core-serving schemas."""
    return '''name: "two-edp-reference-anchored-ranked-model"
index: 0
branch_node {
  updates {
    updates {
      conditional_assignment {
        condition {
          op: HAS
          name: "labeler_input.profile_info.proprietary_id_space_1_user_info.user_id"
        }
        assignments {
          source_field: "labeler_input.profile_info.proprietary_id_space_1_user_info.user_id_fingerprint"
          target_field: "acting_fingerprint"
        }
      }
    }
    updates {
      conditional_assignment {
        condition {
          op: HAS
          name: "labeler_input.profile_info.email_user_info.user_id"
        }
        assignments {
          source_field: "labeler_input.profile_info.email_user_info.user_id_fingerprint"
          target_field: "acting_fingerprint"
        }
      }
    }
  }
  branches {
    condition { op: TRUE }
    node {
      name: "shared-ranked-total-reach-pool"
      index: 1
      ranked_population_node {
        pools {
          population_offset: 10000000
          total_population: 10000
        }
        random_seed: "model-line-2026q3-shared-ranked-pool-v1"
        ranked_size: 9500
        unranked_mode: DISJOINT
      }
    }
  }
}
'''


def allocation_manifest_proto() -> str:
    """Illustrative proposed schema for the aggregate daily instruction."""
    return '''syntax = "proto3";

package calibrated_vid.example;

message EdpReach {
  string edp = 1;
  uint64 reach = 2;
}

// Aggregate control record produced inside the TEE. It contains no raw email,
// Reference ID, proprietary ID, or claimed cross-EDP person link.
message DailyAllocationManifest {
  string model_line = 1;
  uint32 day = 2;
  repeated string edps = 3;
  uint64 requested_cumulative_overlap = 4;
  uint64 achieved_cumulative_overlap = 5;
  repeated EdpReach edp_reaches = 6;
  uint64 union_reach = 7;
  uint64 direct_reference_overlap = 8;
  uint64 synthetic_fallback_overlap = 9;
  string target_status = 10;
  uint64 late_anchor_conflicts = 11;
}
'''


def labeler_input_textproto(event: Impression, assignment: Assignment) -> str:
    profile_lines: list[str] = []
    if event.email is not None:
        profile_lines.extend(
            [
                "  email_user_info {",
                f'    user_id: "{event.email}"',
                "  }",
            ]
        )
    profile_lines.extend(
        [
            "  proprietary_id_space_1_user_info {",
            f'    user_id: "{event.account_id}"',
            "  }",
        ]
    )
    lines = [
        "event_id {",
        f'  publisher: "{event.edp}"',
        f'  id: "{event.event_id}"',
        "}",
        f"timestamp_usec: {1_700_000_000_000_000 + event.day * 86_400_000_000}",
        "profile_info {",
        *profile_lines,
        "}",
        'traffic_info { event_collection_id: "campaign-example" }',
        "rank_assignments {",
        "  pool_offset: 10000000",
        f"  local_rank: {assignment.local_rank}",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _manifest_textproto(summary: DaySummary) -> str:
    return f'''model_line: "2026q3"
day: {summary.day}
edps: "EDP_A"
edps: "EDP_B"
requested_cumulative_overlap: {summary.requested_overlap}
achieved_cumulative_overlap: {summary.achieved_overlap}
edp_reaches {{ edp: "EDP_A" reach: {summary.edp_a_reach} }}
edp_reaches {{ edp: "EDP_B" reach: {summary.edp_b_reach} }}
union_reach: {summary.union_reach}
direct_reference_overlap: {summary.direct_reference_overlap}
synthetic_fallback_overlap: {summary.synthetic_fallback_overlap}
target_status: "{summary.target_status}"
late_anchor_conflicts: {summary.late_anchor_conflicts}
'''


def _walkthrough_markdown(
    allocator: TwoEdpAdaptiveAllocator,
    events: list[Impression],
    feasibility_rows: list[CoverageFeasibility],
) -> str:
    assignment_by_event = {row.event_id: row for row in allocator.assignments}
    lines = [
        "# Two-EDP Reference-ID-anchored VID walkthrough",
        "",
        "Both EDPs run the same static ranked-pool VID model. The model does not change day by day. "
        "The TEE coordinator changes only the rank assigned to each newly observed account, and every "
        "account keeps its first rank forever.",
        "",
        "The 5-billion-value Reference-ID namespace is a join signal, not the VID pool. A matching "
        "Reference ID causes two EDP accounts to receive the same local rank. Proprietary-only accounts "
        "can also share a rank, but only as an aggregate calibration allocation; that sharing does not "
        "assert that the two proprietary IDs identify the same real person.",
        "",
        "| Day | EDP | Account | Reference ID | Rank / VID | Why |",
        "|---:|---|---|---:|---|---|",
    ]
    for event in events:
        row = assignment_by_event[event.event_id]
        reference = str(event.reference_id) if event.reference_id is not None else "—"
        why = row.assignment_reason.replace("_", " ")
        if row.warning:
            why += "; **late anchor conflict flagged**"
        lines.append(
            f"| {event.day} | {event.edp} | `{event.account_id}` | {reference} | "
            f"rank {row.local_rank} → `{row.symbolic_vid}` | {why} |"
        )
    lines.extend(
        [
            "",
            "## Cumulative report state",
            "",
            "| Day | Requested A∩B | Achieved A∩B | Reach A | Reach B | Union | Direct anchors | Synthetic fallback | Status |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for summary in allocator.day_summaries:
        lines.append(
            f"| {summary.day} | {summary.requested_overlap} | {summary.achieved_overlap} | "
            f"{summary.edp_a_reach} | {summary.edp_b_reach} | {summary.union_reach} | "
            f"{summary.direct_reference_overlap} | {summary.synthetic_fallback_overlap} | "
            f"{summary.target_status} |"
        )
    lines.extend(
        [
            "",
            "## What the difficult cases show",
            "",
            "- **Different email coverage works.** On day 2, EDP_B has Eve's email and EDP_A does not. "
            "The Reference-ID rank is fixed at EDP_B, and the residual allocator may place EDP_A's new "
            "proprietary account on that already occupied rank when the calibrated overlap target supports it.",
            "- **Past reports are not loaded.** Stability comes from the persistent account-to-rank and "
            "Reference-ID-to-rank maps. Re-running any old impressions produces the same VIDs.",
            "- **A lower later target cannot erase old overlap.** Day 3 asks for three shared people after "
            "four have already been committed. The output remains four and is flagged as a projection.",
            "- **Late identity evidence can conflict with frozen labels.** Day 5 reveals an email after "
            "Heidi's EDP_A account was already assigned elsewhere. The allocator preserves the old VID, "
            "flags the missed anchor, and can compensate only through other new assignments. This is a "
            "real accuracy limit of immutable online labeling, not a reporting inconsistency.",
            "- **Single-publisher reach is protected.** Within each EDP, no two distinct accounts are put "
            "on the same rank. Cross-EDP sharing changes deduplication, not either publisher's reach.",
            "",
            "## Email-coverage feasibility under the strict design",
            "",
            "This table assumes reach A = 600,000, reach B = 500,000, desired overlap = 300,000, and "
            "60% conditional agreement when both EDPs have email. Only no-email accounts are allowed to "
            "supply synthetic overlap. The result is a capacity check, not an accuracy claim.",
            "",
            "| Email coverage A | Email coverage B | Direct Reference-ID overlap | Flexible A | Flexible B | Maximum reachable overlap | Gap to target |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in feasibility_rows:
        lines.append(
            f"| {row.email_coverage_a:.0%} | {row.email_coverage_b:.0%} | "
            f"{row.direct_reference_overlap:,} | {row.flexible_no_email_a:,} | "
            f"{row.flexible_no_email_b:,} | {row.maximum_reachable_overlap:,} | "
            f"{row.shortfall:,} |"
        )
    lines.extend(
        [
            "",
            "The asymmetric 10%/90% cases remain feasible because the low-coverage EDP supplies a large "
            "flexible residual. The 90%/90% case can fail when conditional email agreement is only 60%: "
            "too many unmatched email-backed ranks are already fixed, while too few no-email accounts "
            "remain to create the missing overlap. Addressing that case requires either better normalized "
            "email agreement, delaying commitment, or allowing unmatched Reference-ID ranks—not only "
            "proprietary fallbacks—to participate in the adaptive allocation. The last option increases "
            "late-anchor conflict risk and needs separate validation.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_example(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events, targets = example_days()
    allocator = TwoEdpAdaptiveAllocator()
    by_day: dict[int, list[Impression]] = {}
    for event in events:
        by_day.setdefault(event.day, []).append(event)
    for day in sorted(by_day):
        allocator.assign_day(day, by_day[day], targets[day])
    feasibility_rows = coverage_sweep()

    (output_dir / "two_edp_ranked_model.textproto").write_text(model_textproto())
    (output_dir / "adaptive_allocation.proto").write_text(allocation_manifest_proto())
    for summary in allocator.day_summaries:
        (output_dir / f"day_{summary.day:02d}_allocation_manifest.textproto").write_text(
            _manifest_textproto(summary)
        )
    event_by_id = {event.event_id: event for event in events}
    input_dir = output_dir / "labeler_inputs"
    input_dir.mkdir(exist_ok=True)
    for assignment in allocator.assignments:
        (input_dir / f"{assignment.event_id}.textproto").write_text(
            labeler_input_textproto(event_by_id[assignment.event_id], assignment)
        )

    with (output_dir / "assignment_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(allocator.assignments[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in allocator.assignments)
    with (output_dir / "email_coverage_feasibility.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(feasibility_rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in feasibility_rows)
    (output_dir / "state.json").write_text(
        json.dumps(
            {
                "account_rank": {
                    f"{edp}:{account}": rank
                    for (edp, account), rank in sorted(allocator.account_rank.items())
                },
                "reference_rank": {
                    str(reference): rank
                    for reference, rank in sorted(allocator.reference_rank.items())
                },
                "reference_edps": {
                    str(reference): sorted(edps)
                    for reference, edps in sorted(allocator.reference_edps.items())
                },
                "rank_edps": {
                    str(rank): sorted(edps) for rank, edps in sorted(allocator.rank_edps.items())
                },
                "day_summaries": [asdict(row) for row in allocator.day_summaries],
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "WALKTHROUGH.md").write_text(
        _walkthrough_markdown(allocator, events, feasibility_rows)
    )
    return {
        "assignments": [asdict(row) for row in allocator.assignments],
        "day_summaries": [asdict(row) for row in allocator.day_summaries],
        "coverage_feasibility": [asdict(row) for row in feasibility_rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/two_edp_adaptive_pools"),
    )
    args = parser.parse_args()
    result = run_example(args.output_dir)
    print(json.dumps(result["day_summaries"], indent=2))


if __name__ == "__main__":
    main()
