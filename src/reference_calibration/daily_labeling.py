from __future__ import annotations

import csv
import heapq
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from math import gcd

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from .config import SimulationConfig
from .measurement import measure_report
from .models import CalibrationModel
from .population import (
    AUDIENCE_STRATEGIES,
    CAMPAIGN_OBJECTIVES,
    META_CAMPAIGN_SCENARIOS,
    Campaign,
    SyntheticWorld,
    generate_campaign,
    make_world,
)
from .sets import exact_cells_from_membership, inclusive_intersections, members, union_values


MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@dataclass(frozen=True)
class DialRegressor:
    """Small provider-trained model for choosing a bridge-pool probability."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    ridge_penalty: float

    def predict(self, features: np.ndarray) -> float:
        standardized = (np.asarray(features, dtype=float) - self.mean) / self.scale
        value = float(np.r_[1.0, standardized] @ self.coefficients)
        return float(np.clip(value, 0.0, 0.95))


@dataclass(frozen=True)
class PairCaptureLogModel(CalibrationModel):
    """Provider-fitted pair capture rates with one shared log-size effect."""

    n_edps: int
    pair_intercepts: np.ndarray
    log_scale_mean: float
    log_scale_slope: float
    name: str = "pair_capture_fixed_plus_log"

    @property
    def pair_list(self) -> tuple[tuple[int, int], ...]:
        return tuple(combinations(range(self.n_edps), 2))

    @property
    def parameter_count(self) -> int:
        return len(self.pair_intercepts) + 1

    def predict_capture(self, subset_masks: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        pair_index = {((1 << left) | (1 << right)): index for index, (left, right) in enumerate(self.pair_list)}
        logits = np.asarray(
            [self.pair_intercepts[pair_index[int(mask)]] for mask in subset_masks],
            dtype=float,
        )
        logits += self.log_scale_slope * (np.asarray(log_scale, dtype=float) - self.log_scale_mean)
        logits = np.clip(logits, -35.0, 35.0)
        return np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-5, 1.0 - 1e-5)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "n_edps": self.n_edps,
            "parameter_count": self.parameter_count,
            "pair_intercepts": self.pair_intercepts.tolist(),
            "log_scale_mean": self.log_scale_mean,
            "log_scale_slope": self.log_scale_slope,
        }


@dataclass(frozen=True)
class LabelingResult:
    method: str
    labels: np.ndarray
    day_dials: np.ndarray
    available_day: int
    notes: str
    supported_edps: int | None = None
    state_entries: int = 0
    pool_count: int = 0
    requires_ordered_days: bool = False


@dataclass(frozen=True)
class DailyExperimentConfig:
    n_users: int = 18_000
    population_size: int = 180_000_000
    n_edps: int = 10
    n_weeks: int = 13
    training_campaigns_per_scenario: int = 2
    evaluation_campaigns_per_scenario: int = 2
    seed: int = 20260829
    bridge_pool_fraction: float = 0.035
    static_bridge_probability: float = 0.12
    warmup_days: int = 3
    dial_grid_size: int = 17
    ridge_penalty: float = 1.0

    @classmethod
    def for_profile(cls, profile: str) -> "DailyExperimentConfig":
        if profile == "quick":
            return cls(
                n_users=6_000,
                training_campaigns_per_scenario=1,
                evaluation_campaigns_per_scenario=1,
                dial_grid_size=11,
            )
        if profile == "full":
            return cls()
        raise ValueError(f"unknown profile: {profile}")


@dataclass(frozen=True)
class PortfolioLabelingResult:
    method: str
    labels_by_campaign: dict[str, np.ndarray]
    state_entries: int
    notes: str


def _mix64(values: np.ndarray | int, seed: int) -> np.ndarray:
    """Vectorized SplitMix64-style deterministic hash."""
    with np.errstate(over="ignore"):
        x = np.asarray(values, dtype=np.uint64) + np.uint64(seed)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x &= MASK64
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x &= MASK64
    return x ^ (x >> np.uint64(31))


def _uniform(values: np.ndarray | int, seed: int) -> np.ndarray:
    hashed = _mix64(values, seed)
    return ((hashed >> np.uint64(11)).astype(np.float64)) / float(1 << 53)


def _identity_keys(
    users: np.ndarray,
    edp: int,
    email_linkable: np.ndarray,
    n_users: int,
) -> tuple[np.ndarray, np.ndarray]:
    email = email_linkable[edp, users]
    # Email keys are common across EDPs. Proprietary keys occupy disjoint EDP
    # namespaces and therefore cannot match except through the VID pool.
    keys = np.where(email, users + 1, (edp + 1) * n_users + users + 1)
    return keys.astype(np.uint64), email


def _labels_for_keys(
    keys: np.ndarray,
    users: np.ndarray,
    edp: int,
    email: np.ndarray,
    dial: float,
    population_size: int,
    bridge_size: int,
) -> np.ndarray:
    """Assign one VID using a stable email anchor and a tunable fallback pool.

    The fallback branch is a two-level nested pool. A stable uniform variate
    decides whether an identifier uses a small shared bridge pool or the full
    population pool. Moving the dial changes only identifiers between the old
    and new thresholds, the minimum possible churn for a Bernoulli branch.
    """
    result = np.empty(len(keys), dtype=np.int64)
    email_users = users[email]
    fallback_keys = keys[~email]
    fallback_users = users[~email]
    if len(email_users):
        # In the synthetic population the normalized email index is already a
        # collision-free global rank. Production would use the 1:1 map and a
        # Feistel permutation, not the literal index.
        result[email] = email_users.astype(np.int64)
    if len(fallback_keys):
        use_bridge = _uniform(fallback_keys, 0xB21D6E) < dial
        multiplier, offset = _affine_parameters(10_000 + edp, population_size)
        raw_rank = (multiplier * fallback_users + offset) % population_size
        second_multiplier, second_offset = _affine_parameters(20_000 + edp, population_size)
        full = ((second_multiplier * raw_rank + second_offset) % population_size).astype(np.int64)
        bridge = (_mix64(raw_rank, 0xB51D) % bridge_size).astype(np.int64)
        result[~email] = np.where(use_bridge, bridge, full)
    return result


def label_hash_pool(
    world: SyntheticWorld,
    campaign: Campaign,
    day_dials: np.ndarray,
    method: str,
    bridge_pool_fraction: float,
    sticky: bool = False,
    available_day: int = 0,
    notes: str = "",
) -> LabelingResult:
    """Label events with a static or day-varying hash/Dirac-like pool."""
    n_edps, n_weeks, n_users = campaign.events.shape
    labels = np.full((n_edps, n_weeks, n_users), -1, dtype=np.int64)
    bridge_size = max(101, int(round(n_users * bridge_pool_fraction)))
    memo: dict[int, int] = {}
    for day in range(n_weeks):
        dial = float(day_dials[day])
        for edp in range(n_edps):
            users = np.flatnonzero(campaign.events[edp, day])
            if not len(users):
                continue
            keys, email = _identity_keys(users, edp, world.email_linkable, n_users)
            assigned = _labels_for_keys(keys, users, edp, email, dial, n_users, bridge_size)
            if sticky:
                for index, key in enumerate(keys.tolist()):
                    if key not in memo:
                        memo[key] = int(assigned[index])
                    assigned[index] = memo[key]
            labels[edp, day, users] = assigned
    return LabelingResult(
        method,
        labels,
        np.asarray(day_dials),
        available_day,
        notes,
        state_entries=len(memo),
        pool_count=2,
        requires_ordered_days=sticky,
    )


def label_collision_resolved_overlap_pool(
    world: SyntheticWorld,
    campaign: Campaign,
    day_dials: np.ndarray,
    bridge_pool_fraction: float,
) -> LabelingResult:
    """Ordered 1:1 map with a tunable cross-EDP collision pool.

    A proposed shared-pool collision is retained across EDPs, but a collision
    with another identifier at the same EDP is deterministically moved to the
    next free VID. This preserves every EDP's marginal reach while keeping the
    overlap dial's intended cross-publisher effect.
    """
    n_edps, n_weeks, n_users = campaign.events.shape
    labels = np.full(campaign.events.shape, -1, dtype=np.int64)
    bridge_size = max(101, int(round(n_users * bridge_pool_fraction)))
    memo: dict[int, int] = {}
    used_by_edp: list[set[int]] = [set() for _ in range(n_edps)]
    reserved_email: set[int] = set()
    globally_used: set[int] = set()
    private_cursor = np.zeros(n_edps, dtype=int)
    globally_free = list(range(n_users))
    heapq.heapify(globally_free)
    raw_parameters = [_affine_parameters(10_000 + edp, n_users) for edp in range(n_edps)]
    full_parameters = [_affine_parameters(20_000 + edp, n_users) for edp in range(n_edps)]

    def choose_slot(edp: int, proposed: int, limit: int) -> int:
        if proposed not in used_by_edp[edp] and proposed not in reserved_email:
            return proposed
        start = int(private_cursor[edp])
        for step in range(n_users):
            candidate = (start + step) % n_users
            if candidate not in used_by_edp[edp] and candidate not in reserved_email:
                private_cursor[edp] = (candidate + 1) % n_users
                return candidate
        return n_users + len(used_by_edp[edp])

    def choose_email_slot(proposed: int) -> int:
        if proposed not in globally_used:
            reserved_email.add(proposed)
            return proposed
        while globally_free:
            candidate = heapq.heappop(globally_free)
            if candidate not in globally_used:
                reserved_email.add(candidate)
                return candidate
        candidate = n_users + len(reserved_email)
        reserved_email.add(candidate)
        return candidate

    for day in range(n_weeks):
        dial = float(day_dials[day])
        for edp in range(n_edps):
            users = np.flatnonzero(campaign.events[edp, day])
            keys, email = _identity_keys(users, edp, world.email_linkable, n_users)
            for key, user, has_email in zip(keys.tolist(), users.tolist(), email.tolist()):
                key = int(key)
                if key not in memo:
                    if has_email:
                        memo[key] = choose_email_slot(int(user))
                    else:
                        use_bridge = float(_uniform(key, 0xB21D6E)[()]) < dial
                        multiplier, offset = raw_parameters[edp]
                        raw_rank = (multiplier * int(user) + offset) % n_users
                        if use_bridge:
                            proposed = int(_mix64(raw_rank, 0xB51D)[()] % np.uint64(bridge_size))
                            memo[key] = choose_slot(edp, proposed, bridge_size)
                        else:
                            second_multiplier, second_offset = full_parameters[edp]
                            proposed = (second_multiplier * raw_rank + second_offset) % n_users
                            memo[key] = choose_slot(edp, proposed, n_users)
                used_by_edp[edp].add(memo[key])
                globally_used.add(memo[key])
                labels[edp, day, user] = memo[key]

    return LabelingResult(
        "ordered_collision_resolved_overlap_pool",
        labels,
        np.asarray(day_dials, dtype=float),
        0,
        "Cumulative match-rate dial with durable mappings and exact per-EDP reach.",
        state_entries=len(memo),
        pool_count=2,
        requires_ordered_days=True,
    )


def label_fixed_marginal_overlap_atlas(
    world: SyntheticWorld,
    campaign: Campaign,
    day_dials: np.ndarray,
) -> LabelingResult:
    """Keep every EDP set size fixed while changing overlap between the sets.

    At each ordered day, the dial selects a cumulative full-roster union
    between the ordinary population-rate union and the maximum-overlap lower
    bound. New identifiers are then assigned either to new slots or to slots
    already occupied by another EDP. Per-EDP reach is exact, the global VID
    space never exceeds the population, and high-affinity EDP memberships are
    preferred when an existing slot is reused.
    """
    n_edps, n_weeks, n_users = campaign.events.shape
    labels = np.full(campaign.events.shape, -1, dtype=np.int64)
    memo: dict[int, int] = {}
    slot_members: dict[int, int] = {}
    slot_first_day: dict[int, int] = {}
    next_vid = 0
    all_edps = tuple(range(n_edps))

    for day in range(n_weeks):
        queues: dict[int, list[int]] = {}
        day_entries: list[tuple[int, np.ndarray, np.ndarray]] = []
        for edp in all_edps:
            users = np.flatnonzero(campaign.events[edp, day])
            keys = ((edp + 1) * n_users + users + 1).astype(np.uint64)
            day_entries.append((edp, users, keys))
            queues[edp] = sorted(
                [int(key) for key in keys.tolist() if int(key) not in memo],
                key=lambda key: int(_mix64(key, 0xF1EED)[()]),
            )

        affinity = _cumulative_pair_affinity(world, campaign, day)
        cumulative_reaches = np.asarray(
            [
                np.any(campaign.events[edp, : day + 1], axis=0).sum()
                for edp in all_edps
            ],
            dtype=float,
        )
        fractions = cumulative_reaches / n_users
        population_rate_union = n_users * (1.0 - float(np.prod(1.0 - fractions)))
        lower = float(np.max(cumulative_reaches))
        upper = float(min(np.sum(cumulative_reaches), n_users))
        target_union = population_rate_union - float(day_dials[day]) * (
            population_rate_union - lower
        )
        target_union = int(round(np.clip(target_union, lower, upper)))
        total_new = sum(len(values) for values in queues.values())
        target_union = int(np.clip(target_union, len(slot_members), len(slot_members) + total_new))
        new_slot_count = target_union - len(slot_members)

        # Create exactly the required number of new slots. Taking from the
        # largest remaining EDP queue keeps the later reuse step feasible.
        for _ in range(new_slot_count):
            edp = max(all_edps, key=lambda item: len(queues[item]))
            if not queues[edp]:
                raise RuntimeError("fixed-marginal allocator ran out of new identifiers")
            key = queues[edp].pop()
            vid = next_vid
            next_vid += 1
            memo[key] = vid
            slot_members[vid] = 1 << edp
            slot_first_day[vid] = day

        # Assign every remaining new identifier to a VID not already present
        # at that EDP. Prefer memberships whose existing EDPs have the highest
        # observed pair affinity, with recent slots as a tie-breaker.
        for edp in all_edps:
            available = [vid for vid, mask in slot_members.items() if not mask & (1 << edp)]
            available.sort(
                key=lambda vid: (
                    max(
                        [affinity[edp, other] for other in members(slot_members[vid], n_edps)]
                        or [0.0]
                    ),
                    slot_first_day[vid],
                    -vid,
                ),
                reverse=True,
            )
            if len(queues[edp]) > len(available):
                raise RuntimeError("fixed-marginal target cannot accommodate an EDP marginal")
            for key, vid in zip(queues[edp], available):
                memo[key] = vid
                slot_members[vid] |= 1 << edp
        for edp, users, keys in day_entries:
            labels[edp, day, users] = np.asarray([memo[int(key)] for key in keys], dtype=np.int64)

    return LabelingResult(
        "fixed_marginal_overlap_atlas",
        labels,
        np.asarray(day_dials, dtype=float),
        0,
        "Fixed EDP marginals; cumulative match signal changes only overlap among their VID sets.",
        state_entries=len(memo),
        pool_count=1,
        requires_ordered_days=True,
    )


def label_pair_targeted_fixed_marginal_atlas(
    world: SyntheticWorld,
    campaign: Campaign,
    capture_model: CalibrationModel,
    union_dials: np.ndarray,
) -> LabelingResult:
    """Preserve every EDP reach while targeting calibrated pair overlaps.

    A provider-trained capture model converts cumulative Reference-ID pair
    matches into target true pair intersections. A separately validated union
    rule chooses the total number of cumulative slots. New identifiers are
    assigned to that fixed number of slots, then remaining identifiers reuse
    existing slots whose EDP memberships have the largest calibrated pair
    deficits. Separating these jobs prevents noisy pair estimates from moving
    the headline total.

    This is deliberately a scalable greedy prototype, not a claim that pairs
    fully identify ten-way overlap. It tests the user's proposed invariant:
    change only cross-EDP pool overlap, never an EDP's pool cardinality.
    """
    n_edps, n_weeks, n_users = campaign.events.shape
    labels = np.full(campaign.events.shape, -1, dtype=np.int64)
    memo: dict[int, int] = {}
    slot_members: dict[int, int] = {}
    slot_first_day: dict[int, int] = {}
    next_vid = 0
    all_edps = tuple(range(n_edps))
    day_targets = np.zeros(n_weeks, dtype=float)

    for day in range(n_weeks):
        queues: dict[int, list[int]] = {}
        day_entries: list[tuple[int, np.ndarray, np.ndarray]] = []
        for edp in all_edps:
            users = np.flatnonzero(campaign.events[edp, day])
            # EDP-local keys make the adjustable fallback channel explicit.
            # A production implementation would first place any directly
            # shared email anchors and apply this allocator to the residual.
            keys = ((edp + 1) * n_users + users + 1).astype(np.uint64)
            day_entries.append((edp, users, keys))
            queues[edp] = sorted(
                [int(key) for key in keys.tolist() if int(key) not in memo],
                key=lambda key: int(_mix64(key, 0xFA17E)[()]),
            )

        observation = measure_report(
            world,
            campaign,
            tuple(range(day + 1)),
            all_edps,
        )
        cumulative_reaches = np.asarray(
            [
                observation.truth_intersections[1 << index] / observation.person_weight
                for index in range(n_edps)
            ],
            dtype=float,
        )
        reach_fractions = cumulative_reaches / n_users
        population_rate_union = n_users * (1.0 - float(np.prod(1.0 - reach_fractions)))
        lower = float(np.max(cumulative_reaches))
        upper = float(min(np.sum(cumulative_reaches), n_users))
        target_union = int(
            round(
                np.clip(
                    population_rate_union
                    - float(union_dials[day]) * (population_rate_union - lower),
                    lower,
                    upper,
                )
            )
        )
        day_targets[day] = float(union_dials[day])
        total_new = sum(len(values) for values in queues.values())
        target_union = int(
            np.clip(target_union, len(slot_members), len(slot_members) + total_new)
        )
        new_slot_count = target_union - len(slot_members)

        for _ in range(new_slot_count):
            edp = max(all_edps, key=lambda item: len(queues[item]))
            if not queues[edp]:
                raise RuntimeError("pair-targeted allocator ran out of new identifiers")
            key = queues[edp].pop()
            vid = next_vid
            next_vid += 1
            memo[key] = vid
            slot_members[vid] = 1 << edp
            slot_first_day[vid] = day

        pair_target = np.zeros((n_edps, n_edps), dtype=float)
        pair_roster = tuple(combinations(all_edps, 2))
        pair_masks = np.asarray(
            [(1 << left) | (1 << right) for left, right in pair_roster],
            dtype=np.int64,
        )
        pair_scales = np.asarray(
            [
                np.log(
                    max(
                        np.sqrt(
                            observation.reach_fractions[left]
                            * observation.reach_fractions[right]
                        ),
                        1e-9,
                    )
                )
                for left, right in pair_roster
            ],
            dtype=float,
        )
        capture = capture_model.predict_capture(pair_masks, pair_scales)
        for (left, right), mask, capture_rate in zip(pair_roster, pair_masks, capture):
            signal = max(float(observation.reference_signal[int(mask)]), 0.0)
            effective = signal / observation.person_weight
            reliability = effective / (effective + 5.0)
            estimate = signal / max(float(capture_rate), 1e-9)
            baseline = float(observation.baseline_intersections[int(mask)])
            lower_pair = max(
                float(observation.truth_intersections[1 << left])
                + float(observation.truth_intersections[1 << right])
                - float(observation.truth_intersections[0]),
                0.0,
            )
            upper_pair = min(
                float(observation.truth_intersections[1 << left]),
                float(observation.truth_intersections[1 << right]),
            )
            value = reliability * estimate + (1.0 - reliability) * baseline
            value = float(np.clip(value, lower_pair, upper_pair)) / observation.person_weight
            pair_target[left, right] = pair_target[right, left] = value

        pair_current = np.zeros((n_edps, n_edps), dtype=float)
        for mask in slot_members.values():
            selected = members(mask, n_edps)
            for left, right in combinations(selected, 2):
                pair_current[left, right] += 1.0
                pair_current[right, left] += 1.0

        # Greedily use slot-sharing opportunities that close the largest
        # normalized pair deficits. Each EDP receives one distinct slot per
        # identifier, so its marginal cardinality cannot change.
        for edp in all_edps:
            available = [vid for vid, mask in slot_members.items() if not mask & (1 << edp)]

            def score(vid: int) -> tuple[float, int, int]:
                other_edps = members(slot_members[vid], n_edps)
                benefit = sum(
                    (pair_target[edp, other] - pair_current[edp, other])
                    / max(pair_target[edp, other], 1.0)
                    for other in other_edps
                )
                return benefit, slot_first_day[vid], -vid

            available.sort(key=score, reverse=True)
            if len(queues[edp]) > len(available):
                raise RuntimeError("pair-targeted target cannot accommodate an EDP marginal")
            for key, vid in zip(queues[edp], available):
                old_members = members(slot_members[vid], n_edps)
                memo[key] = vid
                slot_members[vid] |= 1 << edp
                for other in old_members:
                    pair_current[edp, other] += 1.0
                    pair_current[other, edp] += 1.0

        for edp, users, keys in day_entries:
            labels[edp, day, users] = np.asarray(
                [memo[int(key)] for key in keys],
                dtype=np.int64,
            )

    return LabelingResult(
        "pair_targeted_fixed_marginal_atlas",
        labels,
        day_targets,
        0,
        "Fixed EDP reaches with calibrated pair deficits controlling cross-EDP slot sharing.",
        state_entries=len(memo),
        pool_count=1 + n_edps * (n_edps - 1) // 2,
        requires_ordered_days=True,
    )


def _feature_vector(
    world: SyntheticWorld,
    campaign: Campaign,
    weeks: tuple[int, ...],
    edps: tuple[int, ...],
    feature_set: str = "full",
) -> tuple[np.ndarray, tuple[str, ...]]:
    observation = measure_report(world, campaign, weeks, edps)
    population = float(world.config.population_size)
    reaches = np.asarray(
        [observation.truth_intersections[1 << index] for index in range(len(edps))],
        dtype=float,
    )
    fractions = reaches / population
    values = [
        len(edps) / world.config.n_edps,
        float(np.mean(fractions)),
        float(np.std(fractions)),
        float(np.max(fractions)),
        float(np.log(max(np.exp(np.mean(np.log(np.maximum(fractions, 1e-9)))), 1e-9))),
    ]
    names = [
        "edp_count_fraction",
        "mean_reach_fraction",
        "reach_fraction_std",
        "max_reach_fraction",
        "log_geometric_mean_reach",
    ]

    email_rates = []
    reached = np.any(campaign.events[np.ix_(edps, weeks, np.arange(world.config.n_users))], axis=1)
    for local, edp in enumerate(edps):
        denominator = max(int(reached[local].sum()), 1)
        email_rates.append(float(np.sum(reached[local] & world.email_linkable[edp])) / denominator)
    values.extend((float(np.mean(email_rates)), float(np.std(email_rates))))
    names.extend(("mean_email_availability", "email_availability_std"))

    visible_to_min = []
    visible_to_population_baseline = []
    for left, right in combinations(range(len(edps)), 2):
        mask = (1 << left) | (1 << right)
        visible = max(float(observation.reference_signal[mask]), 0.0)
        minimum = max(min(reaches[left], reaches[right]), observation.person_weight)
        baseline = max(reaches[left] * reaches[right] / population, observation.person_weight)
        visible_to_min.append(visible / minimum)
        visible_to_population_baseline.append(np.log1p(visible / baseline))
    for label, source in (
        ("visible_to_min", visible_to_min),
        ("visible_to_population_baseline", visible_to_population_baseline),
    ):
        array = np.asarray(source, dtype=float)
        values.extend((float(np.mean(array)), float(np.std(array)), float(np.max(array))))
        names.extend((f"{label}_mean", f"{label}_std", f"{label}_max"))

    objectives = tuple(campaign.objectives[edp] for edp in edps)
    strategies = tuple(campaign.audience_strategies[edp] for edp in edps)
    for item in CAMPAIGN_OBJECTIVES:
        values.append(float(np.mean([value == item for value in objectives])))
        names.append(f"objective_{item}")
    for item in AUDIENCE_STRATEGIES:
        values.append(float(np.mean([value == item for value in strategies])))
        names.append(f"strategy_{item}")
    if feature_set == "full":
        selected = np.ones(len(names), dtype=bool)
    elif feature_set == "context_scale":
        selected = np.asarray(
            [
                not (
                    name.startswith("mean_email")
                    or name.startswith("email_availability")
                    or name.startswith("visible_")
                )
                for name in names
            ],
            dtype=bool,
        )
    elif feature_set == "reference_scale":
        selected = np.asarray(
            [not (name.startswith("objective_") or name.startswith("strategy_")) for name in names],
            dtype=bool,
        )
    else:
        raise ValueError(f"unknown feature_set: {feature_set}")
    return np.asarray(values, dtype=float)[selected], tuple(
        name for name, keep in zip(names, selected) if keep
    )


def _report_union(
    labels: np.ndarray,
    weeks: tuple[int, ...],
    edps: tuple[int, ...],
) -> int:
    selected = labels[np.ix_(edps, weeks, np.arange(labels.shape[2]))]
    values = selected[selected >= 0]
    return int(len(np.unique(values)))


def _truth_union(campaign: Campaign, weeks: tuple[int, ...], edps: tuple[int, ...]) -> int:
    reached = np.any(
        campaign.events[np.ix_(edps, weeks, np.arange(campaign.events.shape[2]))],
        axis=(0, 1),
    )
    return int(np.sum(reached))


def _retrospective_dial_loss(
    world: SyntheticWorld,
    campaign: Campaign,
    weeks: tuple[int, ...],
    edps: tuple[int, ...],
    dial: float,
    bridge_pool_fraction: float,
) -> float:
    n_users = campaign.events.shape[2]
    bridge_size = max(101, int(round(n_users * bridge_pool_fraction)))
    reached = np.any(
        campaign.events[np.ix_(edps, weeks, np.arange(n_users))],
        axis=1,
    )
    label_sets: dict[int, set[int]] = {}
    for local, edp in enumerate(edps):
        users = np.flatnonzero(reached[local])
        keys, email = _identity_keys(users, edp, world.email_linkable, n_users)
        label_sets[edp] = set(
            _labels_for_keys(keys, users, edp, email, dial, n_users, bridge_size).tolist()
        )
    # A small fixed roster preserves the pair/medium/full-report trade-off
    # without enumerating all 252 five-EDP subsets of a ten-EDP campaign for
    # every grid point.
    subset_roster = [
        tuple(edps[:2]),
        tuple(edps[-2:]),
        tuple(edps[: min(5, len(edps))]),
        tuple(edps[index] for index in range(0, len(edps), 2))[: min(5, len(edps))],
        tuple(edps),
    ]
    errors = []
    for local_subset in dict.fromkeys(subset_roster):
        if len(local_subset) < 2:
            continue
        truth = _truth_union(campaign, weeks, local_subset)
        estimate = len(set().union(*(label_sets[edp] for edp in local_subset)))
        errors.append(abs(estimate - truth) / max(truth, 1))
    return float(np.mean(errors)) if errors else 0.0


def best_retrospective_dial(
    world: SyntheticWorld,
    campaign: Campaign,
    weeks: tuple[int, ...],
    edps: tuple[int, ...],
    bridge_pool_fraction: float,
    grid_size: int,
) -> float:
    grid = np.linspace(0.0, 0.95, grid_size)
    losses = [
        _retrospective_dial_loss(
            world,
            campaign,
            weeks,
            edps,
            float(dial),
            bridge_pool_fraction,
        )
        for dial in grid
    ]
    return float(grid[int(np.argmin(losses))])


def fit_dial_regressor(
    world: SyntheticWorld,
    campaigns: list[Campaign],
    config: DailyExperimentConfig,
    feature_set: str = "full",
) -> tuple[DialRegressor, list[dict]]:
    features = []
    targets = []
    records = []
    edps = tuple(range(config.n_edps))
    checkpoints = tuple(sorted({0, 2, 5, 8, config.n_weeks - 1}))
    feature_names: tuple[str, ...] | None = None
    for campaign in campaigns:
        for day in checkpoints:
            for basis, weeks in (
                ("same_day", (day,)),
                ("cumulative", tuple(range(day + 1))),
            ):
                vector, names = _feature_vector(
                    world,
                    campaign,
                    weeks,
                    edps,
                    feature_set=feature_set,
                )
                dial = best_retrospective_dial(
                    world,
                    campaign,
                    weeks,
                    edps,
                    config.bridge_pool_fraction,
                    config.dial_grid_size,
                )
                feature_names = names
                features.append(vector)
                targets.append(dial)
                records.append(
                    {
                        "campaign_id": campaign.campaign_id,
                        "scenario": campaign.scenario,
                        "day": day + 1,
                        "basis": basis,
                        "feature_set": feature_set,
                        "best_dial": dial,
                    }
                )
    matrix = np.vstack(features)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (matrix - mean) / scale
    design = np.column_stack([np.ones(len(matrix)), standardized])
    penalty = np.eye(design.shape[1]) * config.ridge_penalty
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ np.asarray(targets, dtype=float),
    )
    model = DialRegressor(
        feature_names or (),
        mean,
        scale,
        coefficients,
        config.ridge_penalty,
    )
    for record, vector in zip(records, features):
        record["predicted_dial"] = model.predict(vector)
        record["absolute_dial_error"] = abs(record["predicted_dial"] - record["best_dial"])
    return model, records


def fit_collision_resolved_regressor(
    world: SyntheticWorld,
    campaigns: list[Campaign],
    config: DailyExperimentConfig,
) -> tuple[DialRegressor, list[dict]]:
    """Fit the overlap dial for the 1:1 collision-resolved encoder itself."""
    features: list[np.ndarray] = []
    targets: list[float] = []
    records: list[dict] = []
    feature_names: tuple[str, ...] | None = None
    edps = tuple(range(config.n_edps))
    checkpoints = tuple(sorted({2, config.n_weeks - 1}))
    grid = np.linspace(0.0, 0.995, min(11, max(9, config.dial_grid_size)))
    subset_roster = (
        (0, 1),
        tuple(range(min(5, config.n_edps))),
        tuple(range(config.n_edps)),
    )
    selected_campaigns = campaigns if len(campaigns) <= 13 else campaigns[::2]
    for campaign in selected_campaigns:
        for day in checkpoints:
            weeks = tuple(range(day + 1))
            vector, names = _feature_vector(world, campaign, weeks, edps, "full")
            losses = []
            for dial in grid:
                labeled = label_collision_resolved_overlap_pool(
                    world,
                    campaign,
                    np.full(config.n_weeks, float(dial)),
                    config.bridge_pool_fraction,
                )
                errors = []
                for subset in subset_roster:
                    truth = _truth_union(campaign, weeks, subset)
                    estimate = _report_union(labeled.labels, weeks, subset)
                    errors.append(abs(estimate - truth) / max(truth, 1))
                losses.append(float(np.mean(errors)))
            target = float(grid[int(np.argmin(losses))])
            feature_names = names
            features.append(vector)
            targets.append(target)
            records.append(
                {
                    "campaign_id": campaign.campaign_id,
                    "scenario": campaign.scenario,
                    "day": day + 1,
                    "basis": "cumulative",
                    "feature_set": "collision_resolved_full",
                    "best_dial": target,
                }
            )
    matrix = np.vstack(features)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack([np.ones(len(matrix)), (matrix - mean) / scale])
    penalty = np.eye(design.shape[1]) * config.ridge_penalty
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ np.asarray(targets, dtype=float),
    )
    model = DialRegressor(
        feature_names or (),
        mean,
        scale,
        coefficients,
        config.ridge_penalty,
    )
    for record, vector in zip(records, features):
        record["predicted_dial"] = model.predict(vector)
        record["absolute_dial_error"] = abs(record["predicted_dial"] - record["best_dial"])
    return model, records


def fit_union_overlap_regressor(
    world: SyntheticWorld,
    campaigns: list[Campaign],
    config: DailyExperimentConfig,
) -> tuple[DialRegressor, list[dict]]:
    """Fit the fixed-marginal fraction between population-rate and minimum union."""
    features: list[np.ndarray] = []
    targets: list[float] = []
    records: list[dict] = []
    edps = tuple(range(config.n_edps))
    checkpoints = tuple(sorted({0, 2, 5, 8, config.n_weeks - 1}))
    feature_names: tuple[str, ...] | None = None
    for campaign in campaigns:
        for day in checkpoints:
            for basis, weeks in (
                ("same_day", (day,)),
                ("cumulative", tuple(range(day + 1))),
            ):
                observation = measure_report(world, campaign, weeks, edps)
                vector, names = _feature_vector(world, campaign, weeks, edps, "full")
                marginals = np.asarray(
                    [observation.truth_intersections[1 << index] for index in range(len(edps))],
                    dtype=float,
                )
                population_rate = float(observation.baseline_unions[-1])
                lower = float(np.max(marginals))
                denominator = max(population_rate - lower, observation.person_weight)
                target = float(
                    np.clip(
                        (population_rate - float(observation.truth_unions[-1])) / denominator,
                        0.0,
                        1.0,
                    )
                )
                feature_names = names
                features.append(vector)
                targets.append(target)
                records.append(
                    {
                        "campaign_id": campaign.campaign_id,
                        "scenario": campaign.scenario,
                        "day": day + 1,
                        "basis": basis,
                        "feature_set": "fixed_marginal_union",
                        "best_dial": target,
                    }
                )
    matrix = np.vstack(features)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack([np.ones(len(matrix)), (matrix - mean) / scale])
    penalty = np.eye(design.shape[1]) * config.ridge_penalty
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ np.asarray(targets, dtype=float),
    )
    model = DialRegressor(
        feature_names or (),
        mean,
        scale,
        coefficients,
        config.ridge_penalty,
    )
    for record, vector in zip(records, features):
        record["predicted_dial"] = model.predict(vector)
        record["absolute_dial_error"] = abs(record["predicted_dial"] - record["best_dial"])
    return model, records


def fit_pair_capture_model(
    world: SyntheticWorld,
    campaigns: list[Campaign],
    config: DailyExperimentConfig,
) -> PairCaptureLogModel:
    """Fit 45 pair capture baselines plus one shared log-size slope at ten EDPs."""
    all_edps = tuple(range(config.n_edps))
    checkpoints = tuple(sorted({0, 2, 5, 8, config.n_weeks - 1}))
    pair_list = tuple(combinations(all_edps, 2))
    rows: list[np.ndarray] = []
    responses: list[float] = []
    weights: list[float] = []
    campaign_ids: list[str] = []
    scales: list[float] = []
    for campaign in campaigns:
        for day in checkpoints:
            observation = measure_report(world, campaign, tuple(range(day + 1)), all_edps)
            for pair_index, (left, right) in enumerate(pair_list):
                mask = (1 << left) | (1 << right)
                truth = float(observation.truth_intersections[mask])
                signal = max(float(observation.reference_signal[mask]), 0.0)
                if truth < observation.person_weight or signal <= 0:
                    continue
                scale = float(
                    np.log(
                        max(
                            np.sqrt(
                                observation.reach_fractions[left]
                                * observation.reach_fractions[right]
                            ),
                            1e-9,
                        )
                    )
                )
                row = np.zeros(len(pair_list), dtype=float)
                row[pair_index] = 1.0
                rows.append(row)
                capture = np.clip(signal / truth, 1e-5, 1.0 - 1e-5)
                responses.append(float(np.log(capture / (1.0 - capture))))
                weights.append(float(np.sqrt(max(signal / observation.person_weight, 1.0))))
                campaign_ids.append(campaign.campaign_id)
                scales.append(scale)
    pair_design = np.vstack(rows)
    scales_array = np.asarray(scales, dtype=float)
    scale_mean = float(np.mean(scales_array))
    design = np.column_stack([pair_design, scales_array - scale_mean])
    weight_array = np.asarray(weights, dtype=float)
    for campaign_id in set(campaign_ids):
        selected = np.asarray([value == campaign_id for value in campaign_ids], dtype=bool)
        norm = float(np.linalg.norm(weight_array[selected]))
        if norm > 0:
            weight_array[selected] /= norm
    weighted_design = design * weight_array[:, None]
    weighted_response = np.asarray(responses, dtype=float) * weight_array
    penalty = np.full(design.shape[1], config.ridge_penalty, dtype=float)
    penalty[-1] = config.ridge_penalty * 0.5
    coefficients, *_ = np.linalg.lstsq(
        np.vstack([weighted_design, np.diag(np.sqrt(penalty))]),
        np.concatenate([weighted_response, np.zeros(design.shape[1])]),
        rcond=None,
    )
    return PairCaptureLogModel(
        n_edps=config.n_edps,
        pair_intercepts=coefficients[:-1],
        log_scale_mean=scale_mean,
        log_scale_slope=float(coefficients[-1]),
    )


def predict_day_dials(
    model: DialRegressor,
    world: SyntheticWorld,
    campaign: Campaign,
    basis: str,
    feature_set: str = "full",
) -> np.ndarray:
    if basis not in {"same_day", "cumulative"}:
        raise ValueError("basis must be same_day or cumulative")
    edps = tuple(range(world.config.n_edps))
    result = np.zeros(world.config.n_weeks, dtype=float)
    for day in range(world.config.n_weeks):
        weeks = (day,) if basis == "same_day" else tuple(range(day + 1))
        vector, names = _feature_vector(
            world,
            campaign,
            weeks,
            edps,
            feature_set=feature_set,
        )
        if names != model.feature_names:
            raise ValueError("dial feature schema changed")
        result[day] = model.predict(vector)
    return result


def _smooth_and_quantize(values: np.ndarray, alpha: float = 0.35, step: float = 0.05) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    state = float(values[0])
    result[0] = round(state / step) * step
    for index in range(1, len(values)):
        state = alpha * float(values[index]) + (1.0 - alpha) * state
        candidate = round(state / step) * step
        # Hysteresis: do not move one quantization step for a marginal signal.
        if abs(candidate - result[index - 1]) >= 1.5 * step:
            result[index] = candidate
        else:
            result[index] = result[index - 1]
    return np.clip(result, 0.0, 0.95)


def _affine_parameters(seed: int, modulus: int) -> tuple[int, int]:
    multiplier = int(_mix64(seed, 0xA771)[()]) % modulus
    multiplier = max(multiplier, 1)
    while gcd(multiplier, modulus) != 1:
        multiplier = (multiplier + 1) % modulus
        if multiplier == 0:
            multiplier = 1
    offset = int(_mix64(seed, 0x0FF5E7)[()]) % modulus
    return multiplier, offset


def _rank_vid(pool_id: int, rank: int, population_size: int) -> int:
    multiplier, offset = _affine_parameters(pool_id + 1, population_size)
    return int((multiplier * rank + offset) % population_size)


def _cumulative_pair_affinity(
    world: SyntheticWorld,
    campaign: Campaign,
    day: int,
) -> np.ndarray:
    edps = tuple(range(world.config.n_edps))
    observation = measure_report(world, campaign, tuple(range(day + 1)), edps)
    affinity = np.zeros((len(edps), len(edps)), dtype=float)
    reaches = np.asarray(
        [observation.truth_intersections[1 << index] for index in range(len(edps))],
        dtype=float,
    )
    for left, right in combinations(range(len(edps)), 2):
        mask = (1 << left) | (1 << right)
        denominator = max(min(reaches[left], reaches[right]), observation.person_weight)
        value = max(float(observation.reference_signal[mask]), 0.0) / denominator
        affinity[left, right] = affinity[right, left] = value
    return affinity


def label_memoized_rank_lattice(
    world: SyntheticWorld,
    campaign: Campaign,
    day_dials: np.ndarray,
    cohort_weeks: int | None = None,
) -> LabelingResult:
    """Stateful 1:1 pool with private, pair, and all-EDP ranked lanes.

    Existing identifiers always reuse their stored VID. Newly seen fallback
    identifiers are routed using the current cumulative match-rate signal.
    Pair lanes align equal local ranks for one EDP pair. Nested prefix lanes
    create coherent three-way and higher-order overlap for unequal EDP sizes.
    With N EDPs this uses N private lanes, N*(N-1)/2 pair lanes, and N-1
    nested lanes: 64 lanes at N=10 rather than 1,023 independent cells.
    """
    n_edps, n_weeks, n_users = campaign.events.shape
    labels = np.full((n_edps, n_weeks, n_users), -1, dtype=np.int64)
    memo: dict[int, int] = {}
    counters: dict[tuple[int, int], int] = {}
    used_by_edp: list[set[int]] = [set() for _ in range(n_edps)]
    globally_used: set[int] = set()
    reserved_shared: set[int] = set()
    shared_rank_vid: dict[tuple[int, int], int] = {}
    private_cursor = np.zeros(n_edps, dtype=int)
    globally_free = list(range(n_users))
    heapq.heapify(globally_free)
    pair_list = tuple(combinations(range(n_edps), 2))
    pair_to_index = {pair: index for index, pair in enumerate(pair_list)}
    prefix_sizes = tuple(range(2, n_edps + 1))
    prefix_to_index = {
        size: len(pair_list) + index for index, size in enumerate(prefix_sizes)
    }
    shared_lane_count = len(pair_list) + len(prefix_sizes)

    def choose_global_slot(proposed: int) -> int:
        if proposed not in globally_used and proposed not in reserved_shared:
            reserved_shared.add(proposed)
            return proposed
        while globally_free:
            candidate = heapq.heappop(globally_free)
            if candidate not in globally_used and candidate not in reserved_shared:
                reserved_shared.add(candidate)
                return candidate
        candidate = n_users + len(reserved_shared)
        reserved_shared.add(candidate)
        return candidate

    def choose_private_slot(edp: int, proposed: int) -> int:
        if proposed not in used_by_edp[edp] and proposed not in reserved_shared:
            return proposed
        start = int(private_cursor[edp])
        for step in range(n_users):
            candidate = (start + step) % n_users
            if candidate not in used_by_edp[edp] and candidate not in reserved_shared:
                private_cursor[edp] = (candidate + 1) % n_users
                return candidate
        return n_users + len(reserved_shared) + len(used_by_edp[edp])

    for day in range(n_weeks):
        affinity = _cumulative_pair_affinity(world, campaign, day)
        shared_probability = min(0.98, 2.1 * float(day_dials[day]))
        cohort = 0 if cohort_weeks is None else day // cohort_weeks
        new_by_route: dict[tuple[int, int], list[int]] = {}
        day_users: list[tuple[int, np.ndarray, np.ndarray]] = []
        for edp in range(n_edps):
            users = np.flatnonzero(campaign.events[edp, day])
            keys, email = _identity_keys(users, edp, world.email_linkable, n_users)
            day_users.append((edp, users, keys))
            for key, user in zip(keys[email].tolist(), users[email].tolist()):
                if int(key) not in memo:
                    memo[int(key)] = choose_global_slot(int(user))
                used_by_edp[edp].add(memo[int(key)])
                globally_used.add(memo[int(key)])

            new_fallback = [int(key) for key in keys[~email].tolist() if int(key) not in memo]
            if not new_fallback:
                continue
            other_edps = [other for other in range(n_edps) if other != edp]
            pair_weights = np.asarray([affinity[edp, other] + 0.02 for other in other_edps])
            pair_weights /= float(pair_weights.sum())
            for key in new_fallback:
                shared = float(_uniform(key, 0x511A2E)[()]) < shared_probability
                if not shared:
                    pool_id = edp
                else:
                    # Nested prefix lanes let one assignment create several
                    # coherent pair overlaps. Pair lanes retain local affinity
                    # that a single all-EDP pool cannot express.
                    structure_draw = float(_uniform(key, 0x610BA1)[()])
                    eligible_prefixes = [size for size in prefix_sizes if edp < size]
                    if structure_draw < 0.62 and eligible_prefixes:
                        prefix_draw = float(_uniform(key, 0xC0A0A7)[()])
                        # Prefer the smallest eligible nested group, with a
                        # long tail toward the full roster.
                        weights = 1.0 / np.asarray(eligible_prefixes, dtype=float)
                        weights /= float(weights.sum())
                        selected = eligible_prefixes[
                            min(
                                int(np.searchsorted(np.cumsum(weights), prefix_draw)),
                                len(eligible_prefixes) - 1,
                            )
                        ]
                        shared_index = prefix_to_index[selected]
                    else:
                        draw = float(_uniform(key, 0xA7712)[()])
                        cumulative = np.cumsum(pair_weights)
                        other = other_edps[min(int(np.searchsorted(cumulative, draw)), len(other_edps) - 1)]
                        shared_index = pair_to_index[tuple(sorted((edp, other)))]
                    pool_id = n_edps + cohort * shared_lane_count + shared_index
                new_by_route.setdefault((edp, pool_id), []).append(key)

        for (edp, pool_id), keys in sorted(new_by_route.items()):
            ordered = sorted(keys, key=lambda key: int(_mix64(key, 0x52A6E)[()]))
            counter_key = (edp, pool_id)
            start = counters.get(counter_key, 0)
            for offset, key in enumerate(ordered):
                rank = start + offset
                proposed = _rank_vid(pool_id, rank, n_users)
                if pool_id == edp:
                    vid = choose_private_slot(edp, proposed)
                else:
                    slot_key = (pool_id, rank)
                    if slot_key not in shared_rank_vid:
                        shared_rank_vid[slot_key] = choose_global_slot(proposed)
                    vid = shared_rank_vid[slot_key]
                    if vid in used_by_edp[edp]:
                        # This can happen only after the fixed population is
                        # saturated. Preserve the EDP marginal rather than
                        # silently assigning two local IDs to one VID.
                        vid = choose_private_slot(edp, proposed)
                memo[key] = vid
                used_by_edp[edp].add(vid)
                globally_used.add(vid)
            counters[counter_key] = start + len(ordered)

        for edp, users, keys in day_users:
            labels[edp, day, users] = np.asarray([memo[int(key)] for key in keys], dtype=np.int64)

    method = (
        "ordered_memoized_rank_lattice"
        if cohort_weeks is None
        else f"cohort_{cohort_weeks}_week_rank_lattice"
    )
    cohort_count = 1 if cohort_weeks is None else int(np.ceil(n_weeks / cohort_weeks))
    return LabelingResult(
        method,
        labels,
        np.asarray(day_dials, dtype=float),
        0,
        (
            "Sequential 1:1 map with private, pair, and nested higher-order ranked lanes."
            if cohort_weeks is None
            else "Shared ranked lanes are repeated by first-seen time cohort to reduce cross-window distortion."
        ),
        state_entries=len(memo),
        pool_count=n_edps + shared_lane_count * cohort_count,
        requires_ordered_days=True,
    )


def _transport_cells(current: np.ndarray, target: np.ndarray) -> dict[tuple[int, int], int]:
    """Transport current Venn cells into cumulative target cells by only adding memberships."""
    n = int(round(np.log2(len(target))))
    current = np.rint(current).astype(int)
    target = np.rint(target).astype(int)
    union_growth = int(target[1:].sum() - current[1:].sum())
    if union_growth < 0:
        raise ValueError("cumulative target union cannot shrink")
    supplies = {0: union_growth}
    supplies.update({mask: int(current[mask]) for mask in range(1, 1 << n) if current[mask]})
    demands = {mask: int(target[mask]) for mask in range(1, 1 << n) if target[mask]}
    edges = [
        (source, destination)
        for source in supplies
        for destination in demands
        if source == 0 or source & destination == source
    ]
    if not edges:
        return {}
    source_rows = {mask: index for index, mask in enumerate(supplies)}
    destination_rows = {
        mask: len(source_rows) + index for index, mask in enumerate(demands)
    }
    matrix = np.zeros((len(source_rows) + len(destination_rows), len(edges)), dtype=float)
    for column, (source, destination) in enumerate(edges):
        matrix[source_rows[source], column] = 1.0
        matrix[destination_rows[destination], column] = 1.0
    rhs = np.asarray(list(supplies.values()) + list(demands.values()), dtype=float)
    costs = np.asarray(
        [
            0.01 * (destination.bit_count() - source.bit_count())
            + 0.0001 * destination.bit_count()
            for source, destination in edges
        ],
        dtype=float,
    )
    solved = linprog(costs, A_eq=matrix, b_eq=rhs, bounds=(0.0, None), method="highs")
    if not solved.success:
        raise RuntimeError(f"online Venn transport is infeasible: {solved.message}")
    rounded = np.rint(solved.x).astype(int)
    if np.max(np.abs(matrix @ rounded - rhs)) > 1e-6:
        raise RuntimeError("online Venn transport did not produce an integral flow")
    return {
        edge: int(count)
        for edge, count in zip(edges, rounded)
        if count > 0
    }


def label_oracle_online_venn(
    world: SyntheticWorld,
    campaign: Campaign,
    edp_count: int = 5,
    prefer_recent_slots: bool = True,
) -> LabelingResult:
    """Online exact-cell pool used as a structural, not deployable, upper bound.

    It is intentionally given the true cumulative Venn cells. The experiment
    asks whether an immutable sequential map could encode those cells, and how
    much error remains for arbitrary subwindows when the allocator cannot know
    future arrival times. Production would replace truth with a provider-owned
    prediction from approved aggregates.
    """
    n_weeks, n_users = campaign.events.shape[1], campaign.events.shape[2]
    labels = np.full(campaign.events.shape, -1, dtype=np.int64)
    memo: dict[int, int] = {}
    slot_members: dict[int, int] = {}
    slot_first_day: dict[int, int] = {}
    next_synthetic_vid = n_users
    tracked_edps = tuple(range(edp_count))

    for day in range(n_weeks):
        day_entries: list[tuple[int, np.ndarray, np.ndarray]] = []
        new_fallback: dict[int, list[int]] = {edp: [] for edp in tracked_edps}
        for edp in tracked_edps:
            users = np.flatnonzero(campaign.events[edp, day])
            # The aggregate-only upper bound deliberately uses an EDP-local
            # stable key even when email is present. It therefore proves what
            # a coordinated synthetic pool can encode without assuming that a
            # proprietary ID can later be linked to a common email identity.
            keys = ((edp + 1) * n_users + users + 1).astype(np.uint64)
            day_entries.append((edp, users, keys))
            for key in keys.tolist():
                key = int(key)
                if key not in memo:
                    new_fallback[edp].append(key)

        current = np.zeros(1 << edp_count, dtype=int)
        for mask in slot_members.values():
            current[mask] += 1
        reached = np.any(
            campaign.events[np.ix_(tracked_edps, tuple(range(day + 1)), np.arange(n_users))],
            axis=1,
        )
        target = exact_cells_from_membership(reached, 1.0).astype(int)
        flow = _transport_cells(current, target)
        slots_by_mask: dict[int, list[int]] = {}
        for vid, mask in slot_members.items():
            slots_by_mask.setdefault(mask, []).append(vid)
        for values in slots_by_mask.values():
            values.sort(
                key=lambda vid: (slot_first_day[vid], vid),
                reverse=prefer_recent_slots,
            )
        ordered_new = {
            edp: sorted(keys, key=lambda key: int(_mix64(key, 0x0A11CE)[()]))
            for edp, keys in new_fallback.items()
        }

        for (source, destination), count in sorted(flow.items()):
            if source == 0:
                selected_vids = list(range(next_synthetic_vid, next_synthetic_vid + count))
                next_synthetic_vid += count
                for vid in selected_vids:
                    slot_first_day[vid] = day
            else:
                selected_vids = slots_by_mask[source][:count]
                del slots_by_mask[source][:count]
            additions = [edp for edp in tracked_edps if destination & (1 << edp) and not source & (1 << edp)]
            for vid in selected_vids:
                for edp in additions:
                    if not ordered_new[edp]:
                        raise RuntimeError("Venn flow consumed more new identifiers than available")
                    key = ordered_new[edp].pop()
                    memo[key] = vid
                slot_members[vid] = destination

        if any(ordered_new[edp] for edp in tracked_edps):
            raise RuntimeError("Venn flow left new fallback identifiers unassigned")
        for edp, users, keys in day_entries:
            labels[edp, day, users] = np.asarray([memo[int(key)] for key in keys], dtype=np.int64)

    label = "oracle_online_venn_recent" if prefer_recent_slots else "oracle_online_venn_oldest"
    return LabelingResult(
        label,
        labels,
        np.zeros(n_weeks, dtype=float),
        0,
        "Uses true cumulative five-EDP cells only as an upper bound; stored IDs never move.",
        supported_edps=edp_count,
        state_entries=len(memo),
        pool_count=(1 << edp_count) - 1,
        requires_ordered_days=True,
    )


def label_oracle_online_union(
    world: SyntheticWorld,
    campaign: Campaign,
    prefer_recent_slots: bool = True,
) -> LabelingResult:
    """Scalable upper bound that targets only the full-roster cumulative union.

    Every EDP-local identifier keeps one stored VID and every EDP marginal is
    exact. The allocator creates exactly enough new slots to hit the desired
    all-EDP cumulative union, then places remaining new IDs into slots already
    occupied by another EDP. It does not target pairwise subset geometry.
    """
    n_edps, n_weeks, n_users = campaign.events.shape
    labels = np.full(campaign.events.shape, -1, dtype=np.int64)
    memo: dict[int, int] = {}
    slot_members: dict[int, int] = {}
    slot_first_day: dict[int, int] = {}
    next_vid = 0
    all_edps = tuple(range(n_edps))

    for day in range(n_weeks):
        day_entries: list[tuple[int, np.ndarray, np.ndarray]] = []
        new_keys: dict[int, list[int]] = {}
        for edp in all_edps:
            users = np.flatnonzero(campaign.events[edp, day])
            keys = ((edp + 1) * n_users + users + 1).astype(np.uint64)
            day_entries.append((edp, users, keys))
            new_keys[edp] = sorted(
                [int(key) for key in keys.tolist() if int(key) not in memo],
                key=lambda key: int(_mix64(key, 0xA110C)[()]),
            )

        cumulative_truth = _truth_union(
            campaign,
            tuple(range(day + 1)),
            all_edps,
        )
        new_slot_count = cumulative_truth - len(slot_members)
        if new_slot_count < 0 or new_slot_count > sum(map(len, new_keys.values())):
            raise RuntimeError("full-union target is outside the online allocation bounds")

        # Give each required new slot one new identifier. Choosing the EDP
        # with the largest remaining queue keeps the later reuse step feasible
        # for asymmetric campaign sizes.
        for _ in range(new_slot_count):
            edp = max(all_edps, key=lambda item: len(new_keys[item]))
            if not new_keys[edp]:
                raise RuntimeError("not enough new identifiers to create required union slots")
            key = new_keys[edp].pop()
            vid = next_vid
            next_vid += 1
            memo[key] = vid
            slot_members[vid] = 1 << edp
            slot_first_day[vid] = day

        for edp in all_edps:
            available = [vid for vid, mask in slot_members.items() if not mask & (1 << edp)]
            available.sort(
                key=lambda vid: (slot_first_day[vid], vid),
                reverse=prefer_recent_slots,
            )
            if len(new_keys[edp]) > len(available):
                raise RuntimeError("not enough cross-EDP slots to preserve the marginal")
            for key, vid in zip(new_keys[edp], available):
                memo[key] = vid
                slot_members[vid] |= 1 << edp

        for edp, users, keys in day_entries:
            labels[edp, day, users] = np.asarray([memo[int(key)] for key in keys], dtype=np.int64)

    method = "oracle_online_union_recent" if prefer_recent_slots else "oracle_online_union_oldest"
    return LabelingResult(
        method,
        labels,
        np.zeros(n_weeks, dtype=float),
        0,
        "Uses true cumulative all-EDP union only; exact full-roster prefixes but no subset target.",
        state_entries=len(memo),
        pool_count=1,
        requires_ordered_days=True,
    )


def label_oracle_person_identity(campaign: Campaign) -> LabelingResult:
    """Forbidden identity-graph oracle: every event for one real person gets one VID."""
    n_edps, n_weeks, n_users = campaign.events.shape
    labels = np.full(campaign.events.shape, -1, dtype=np.int64)
    for edp in range(n_edps):
        for day in range(n_weeks):
            users = np.flatnonzero(campaign.events[edp, day])
            labels[edp, day, users] = users
    return LabelingResult(
        "forbidden_person_identity_oracle",
        labels,
        np.zeros(n_weeks, dtype=float),
        0,
        "Accuracy ceiling only; requires the cross-EDP identity graph that the design forbids.",
        state_entries=n_users,
        pool_count=1,
        requires_ordered_days=False,
    )


def build_labeling_methods(
    model: DialRegressor,
    context_model: DialRegressor,
    resolved_model: DialRegressor,
    union_model: DialRegressor,
    pair_capture_model: PairCaptureLogModel,
    world: SyntheticWorld,
    campaign: Campaign,
    config: DailyExperimentConfig,
    static_dial: float,
) -> list[LabelingResult]:
    same_day = predict_day_dials(model, world, campaign, "same_day", "full")
    cumulative = predict_day_dials(model, world, campaign, "cumulative", "full")
    context_only = predict_day_dials(
        context_model,
        world,
        campaign,
        "cumulative",
        "context_scale",
    )
    smoothed = _smooth_and_quantize(cumulative)
    blended = 0.55 * cumulative + 0.45 * context_only
    resolved = predict_day_dials(
        resolved_model,
        world,
        campaign,
        "cumulative",
        "full",
    )
    union_dial = predict_day_dials(
        union_model,
        world,
        campaign,
        "cumulative",
        "full",
    )
    static = np.full(config.n_weeks, static_dial, dtype=float)
    frozen = np.full(config.n_weeks, cumulative[config.warmup_days - 1], dtype=float)
    full_flight_frozen = np.full(config.n_weeks, cumulative[-1], dtype=float)
    oracle = np.asarray(
        [
            best_retrospective_dial(
                world,
                campaign,
                (day,),
                tuple(range(config.n_edps)),
                config.bridge_pool_fraction,
                config.dial_grid_size,
            )
            for day in range(config.n_weeks)
        ],
        dtype=float,
    )
    return [
        label_hash_pool(
            world,
            campaign,
            static,
            "fixed_model_line_pool",
            config.bridge_pool_fraction,
            notes="One model-line configuration; no daily adaptation.",
        ),
        label_hash_pool(
            world,
            campaign,
            context_only,
            "context_and_scale_only_pool",
            config.bridge_pool_fraction,
            sticky=True,
            notes="Ordered 1:1 map uses objective, audience strategy, and scale but no observed Reference-ID overlap.",
        ),
        label_hash_pool(
            world,
            campaign,
            blended,
            "provider_blended_ordered_pool",
            config.bridge_pool_fraction,
            sticky=True,
            notes="Ordered 1:1 map blends the campaign-context prior with cumulative observed matching.",
        ),
        label_collision_resolved_overlap_pool(
            world,
            campaign,
            resolved,
            config.bridge_pool_fraction,
        ),
        label_fixed_marginal_overlap_atlas(world, campaign, union_dial),
        label_pair_targeted_fixed_marginal_atlas(
            world,
            campaign,
            pair_capture_model,
            union_dial,
        ),
        label_hash_pool(
            world,
            campaign,
            same_day,
            "same_day_adaptive_pool",
            config.bridge_pool_fraction,
            notes="Uses only that day's aggregate signal; labels are final immediately.",
        ),
        label_hash_pool(
            world,
            campaign,
            cumulative,
            "cumulative_adaptive_pool",
            config.bridge_pool_fraction,
            notes="Uses all data observed through the labeling day.",
        ),
        label_hash_pool(
            world,
            campaign,
            smoothed,
            "quantized_hysteresis_pool",
            config.bridge_pool_fraction,
            notes="Cumulative estimate is smoothed and changed only in coarse steps.",
        ),
        label_hash_pool(
            world,
            campaign,
            cumulative,
            "sticky_first_seen_pool",
            config.bridge_pool_fraction,
            sticky=True,
            notes="The current dial affects only an identifier's first appearance.",
        ),
        label_memoized_rank_lattice(world, campaign, smoothed),
        label_memoized_rank_lattice(world, campaign, smoothed, cohort_weeks=3),
        label_oracle_online_venn(world, campaign, prefer_recent_slots=True),
        label_oracle_online_venn(world, campaign, prefer_recent_slots=False),
        label_oracle_online_union(world, campaign, prefer_recent_slots=True),
        label_oracle_online_union(world, campaign, prefer_recent_slots=False),
        label_oracle_person_identity(campaign),
        label_hash_pool(
            world,
            campaign,
            frozen,
            "three_day_buffer_then_freeze",
            config.bridge_pool_fraction,
            available_day=config.warmup_days,
            notes="Days 1-3 are buffered, then all days use one flight-level setting.",
        ),
        label_hash_pool(
            world,
            campaign,
            full_flight_frozen,
            "full_flight_buffer_then_freeze",
            config.bridge_pool_fraction,
            available_day=config.n_weeks,
            notes="All events are buffered until the full-flight cumulative signal is known.",
        ),
        label_hash_pool(
            world,
            campaign,
            oracle,
            "oracle_same_day_dial",
            config.bridge_pool_fraction,
            notes="Unimplementable upper bound: each day uses its best retrospective dial.",
        ),
    ]


def fit_static_dial(
    world: SyntheticWorld,
    campaigns: list[Campaign],
    config: DailyExperimentConfig,
) -> float:
    broad = [campaign for campaign in campaigns if campaign.scenario == "broad_awareness_control"]
    selected = broad or campaigns
    weeks = tuple(range(config.n_weeks))
    edps = tuple(range(config.n_edps))
    grid = np.linspace(0.0, 0.95, config.dial_grid_size)
    losses = []
    for dial in grid:
        losses.append(
            float(
                np.mean(
                    [
                        _retrospective_dial_loss(
                            world,
                            campaign,
                            weeks,
                            edps,
                            float(dial),
                            config.bridge_pool_fraction,
                        )
                        for campaign in selected
                    ]
                )
            )
        )
    return float(grid[int(np.argmin(losses))])


def report_specs(n_edps: int, n_weeks: int) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]:
    all_edps = tuple(range(n_edps))
    five = tuple(range(min(5, n_edps)))
    return (
        ("weeks_1_3__2_edps", tuple(range(3)), (0, 1)),
        ("weeks_1_12__2_edps", tuple(range(min(12, n_weeks))), (0, 1)),
        ("weeks_5_12__2_edps", tuple(range(4, min(12, n_weeks))), (0, 1)),
        ("weeks_1_3__5_edps", tuple(range(3)), five),
        ("weeks_1_12__5_edps", tuple(range(min(12, n_weeks))), five),
        ("weeks_7_13__5_edps", tuple(range(6, n_weeks)), five),
        ("all_weeks__5_edps", tuple(range(n_weeks)), five),
        (
            "noncontiguous__5_edps",
            tuple(index for index in (0, 2, 4, 7, 10, 12) if index < n_weeks),
            five,
        ),
        ("weeks_1_3__10_edps", tuple(range(3)), all_edps),
        ("weeks_1_12__10_edps", tuple(range(min(12, n_weeks))), all_edps),
        ("all_weeks__10_edps", tuple(range(n_weeks)), all_edps),
    )


def _fragmentation_metrics(
    world: SyntheticWorld,
    campaign: Campaign,
    result: LabelingResult,
) -> tuple[float, float, float]:
    local_key_labels: dict[tuple[int, int], set[int]] = {}
    local_key_events: dict[tuple[int, int], int] = {}
    email_labels: dict[int, set[int]] = {}
    email_edps: dict[int, set[int]] = {}
    person_labels: dict[int, set[int]] = {}
    n_edps, n_weeks, n_users = campaign.events.shape
    for edp in range(n_edps):
        for day in range(n_weeks):
            users = np.flatnonzero(campaign.events[edp, day])
            if not len(users):
                continue
            keys, _ = _identity_keys(users, edp, world.email_linkable, n_users)
            labels = result.labels[edp, day, users]
            email = world.email_linkable[edp, users]
            for key, user, label, has_email in zip(
                keys.tolist(), users.tolist(), labels.tolist(), email.tolist()
            ):
                local_key = (edp, int(key))
                local_key_labels.setdefault(local_key, set()).add(int(label))
                local_key_events[local_key] = local_key_events.get(local_key, 0) + 1
                if has_email:
                    email_labels.setdefault(int(user), set()).add(int(label))
                    email_edps.setdefault(int(user), set()).add(edp)
                person_labels.setdefault(int(user), set()).add(int(label))
    repeated_keys = [
        local_key_labels[key] for key, count in local_key_events.items() if count > 1
    ]
    key_fragmentation = (
        float(np.mean([len(values) > 1 for values in repeated_keys]))
        if repeated_keys
        else 0.0
    )
    cross_edp_emails = [
        email_labels[user] for user, edps in email_edps.items() if len(edps) > 1
    ]
    email_fragmentation = (
        float(np.mean([len(values) > 1 for values in cross_edp_emails]))
        if cross_edp_emails
        else 0.0
    )
    person_multiplicity = float(np.mean([len(values) for values in person_labels.values()])) if person_labels else 0.0
    return key_fragmentation, email_fragmentation, person_multiplicity


def _intersection_errors(
    campaign: Campaign,
    labels: np.ndarray,
    weeks: tuple[int, ...],
    edps: tuple[int, ...],
) -> dict[int, dict[str, float]]:
    users = np.arange(campaign.events.shape[2])
    truth_membership = np.any(campaign.events[np.ix_(edps, weeks, users)], axis=1)
    truth_cells = exact_cells_from_membership(truth_membership, 1.0)
    truth_intersections = inclusive_intersections(truth_cells)
    label_sets = []
    for edp in edps:
        values = labels[np.ix_((edp,), weeks, users)].reshape(-1)
        label_sets.append(set(values[values >= 0].tolist()))
    label_membership: dict[int, int] = {}
    for local, values in enumerate(label_sets):
        bit = 1 << local
        for value in values:
            label_membership[value] = label_membership.get(value, 0) | bit
    estimated_cells = np.bincount(
        np.fromiter(label_membership.values(), dtype=np.int64),
        minlength=1 << len(edps),
    ).astype(float)
    estimated_intersections = inclusive_intersections(estimated_cells)
    relative: dict[int, list[float]] = {}
    absolute_total: dict[int, float] = {}
    truth_total: dict[int, float] = {}
    for mask in range(1, 1 << len(edps)):
        order = mask.bit_count()
        if order < 2:
            continue
        estimate = float(estimated_intersections[mask])
        truth = float(truth_intersections[mask])
        error = abs(estimate - truth)
        relative.setdefault(order, []).append(error / max(truth, 1.0))
        absolute_total[order] = absolute_total.get(order, 0.0) + error
        truth_total[order] = truth_total.get(order, 0.0) + truth
    return {
        order: {
            "mean_relative": float(np.mean(values)),
            "weighted_relative": absolute_total[order] / max(truth_total[order], 1.0),
            "absolute_total": absolute_total[order],
            "truth_total": truth_total[order],
        }
        for order, values in relative.items()
    }


def evaluate_labeling(
    world: SyntheticWorld,
    campaign: Campaign,
    result: LabelingResult,
) -> list[dict]:
    key_fragmentation, email_fragmentation, person_multiplicity = _fragmentation_metrics(
        world,
        campaign,
        result,
    )
    churn = float(np.mean(np.abs(np.diff(result.day_dials)))) if len(result.day_dials) > 1 else 0.0
    rows = []
    for report_name, weeks, edps in report_specs(world.config.n_edps, world.config.n_weeks):
        if result.supported_edps is not None and any(edp >= result.supported_edps for edp in edps):
            continue
        truth = _truth_union(campaign, weeks, edps)
        estimate = _report_union(result.labels, weeks, edps)
        intersection = _intersection_errors(campaign, result.labels, weeks, edps)
        marginal_errors = []
        for edp in edps:
            marginal_truth = _truth_union(campaign, weeks, (edp,))
            marginal_estimate = _report_union(result.labels, weeks, (edp,))
            marginal_errors.append(
                abs(marginal_estimate - marginal_truth) / max(marginal_truth, 1)
            )
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
                "edp_count": len(edps),
                "week_count": len(weeks),
                "report_type": report_type,
                "truth_union": truth,
                "estimated_union": estimate,
                "union_relative_error": abs(estimate - truth) / max(truth, 1),
                "signed_union_error": (estimate - truth) / max(truth, 1),
                "population_bound_excess": max(estimate - world.config.n_users, 0)
                / world.config.n_users,
                "mean_marginal_relative_error": float(np.mean(marginal_errors)),
                "pair_intersection_error": float(
                    intersection.get(2, {}).get("weighted_relative", 0.0)
                ),
                "three_way_intersection_error": float(
                    intersection.get(3, {}).get("weighted_relative", 0.0)
                ),
                "four_plus_intersection_error": float(
                    sum(
                        values["absolute_total"]
                        for order, values in intersection.items()
                        if order >= 4
                    )
                    / max(
                        sum(
                            values["truth_total"]
                            for order, values in intersection.items()
                            if order >= 4
                        ),
                        1.0,
                    )
                ),
                "stable_key_fragmentation": key_fragmentation,
                "cross_edp_email_fragmentation": email_fragmentation,
                "mean_vids_per_true_person": person_multiplicity,
                "mean_day_dial": float(np.mean(result.day_dials)),
                "day_dial_churn": churn,
                "first_available_day": result.available_day,
                "state_entries": result.state_entries,
                "pool_count": result.pool_count,
                "requires_ordered_days": result.requires_ordered_days,
                "notes": result.notes,
            }
        )
    return rows


def _consistency_audit(campaign: Campaign, result: LabelingResult) -> dict[str, int | float]:
    specs = tuple(
        spec
        for spec in report_specs(campaign.events.shape[0], campaign.events.shape[1])
        if result.supported_edps is None or all(edp < result.supported_edps for edp in spec[2])
    )
    values = {
        name: (_report_union(result.labels, weeks, edps), set(edps), set(weeks))
        for name, weeks, edps in specs
    }
    checks = violations = 0
    max_violation = 0
    items = list(values.items())
    for _, (left_value, left_edps, left_weeks) in items:
        for _, (right_value, right_edps, right_weeks) in items:
            if left_edps <= right_edps and left_weeks <= right_weeks:
                checks += 1
                violation = left_value - right_value
                if violation > 0:
                    violations += 1
                    max_violation = max(max_violation, violation)
    early = _report_union(result.labels, tuple(range(3)), (0, 1))
    early_recomputed = _report_union(result.labels, tuple(range(3)), (0, 1))
    return {
        "nested_report_checks": checks,
        "nested_report_violations": violations,
        "max_nested_violation": max_violation,
        "weeks_1_3_replay_difference": abs(early - early_recomputed),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)) if len(array) else 0.0,
        "median": float(np.median(array)) if len(array) else 0.0,
        "p90": float(np.quantile(array, 0.9)) if len(array) else 0.0,
        "max": float(np.max(array)) if len(array) else 0.0,
    }


def generate_temporal_stress_campaign(
    world: SyntheticWorld,
    kind: str,
    seed: int,
    campaign_id: str,
) -> Campaign:
    """Create plausible retargeting flights with deliberately difficult timing."""
    rng = np.random.default_rng(seed)
    n_edps, n_weeks, n_users = (
        world.config.n_edps,
        world.config.n_weeks,
        world.config.n_users,
    )
    events = np.zeros((n_edps, n_weeks, n_users), dtype=bool)
    shared_size = max(10, int(round(0.09 * n_users)))
    shared = np.argpartition(world.segments[0] + 0.25 * world.matchability, -shared_size)[
        -shared_size:
    ]
    remaining = np.setdiff1d(np.arange(n_users), shared, assume_unique=False)
    unique_size = max(5, int(round(0.018 * n_users)))

    for edp in range(n_edps):
        selected_shared = rng.choice(shared, size=int(0.82 * shared_size), replace=False)
        selected_unique = rng.choice(remaining, size=unique_size, replace=False)
        if kind == "staggered_retargeting":
            center = int(round(edp * (n_weeks - 1) / max(n_edps - 1, 1)))
            shared_days = np.clip(
                center + rng.integers(-1, 2, size=len(selected_shared)),
                0,
                n_weeks - 1,
            )
        elif kind == "synchronized_retargeting":
            shared_days = np.clip(
                5 + rng.integers(-1, 2, size=len(selected_shared)),
                0,
                n_weeks - 1,
            )
        elif kind == "shared_seed_then_expansion":
            shared_days = rng.integers(0, min(3, n_weeks), size=len(selected_shared))
            selected_unique = rng.choice(
                remaining,
                size=max(unique_size, int(round(0.10 * n_users))),
                replace=False,
            )
        else:
            raise ValueError(f"unknown temporal stress kind: {kind}")
        unique_days = rng.integers(max(0, n_weeks - 5), n_weeks, size=len(selected_unique))
        events[edp, shared_days, selected_shared] = True
        events[edp, unique_days, selected_unique] = True

    return Campaign(
        campaign_id=campaign_id,
        scenario=kind,
        events=events,
        final_reach_fraction=np.any(events, axis=1).mean(axis=1),
        objectives=tuple("sales" for _ in range(n_edps)),
        audience_strategies=tuple("website_retargeting" for _ in range(n_edps)),
    )


def label_campaign_portfolio(
    world: SyntheticWorld,
    campaigns: list[Campaign],
    dials: dict[str, np.ndarray],
    method: str,
    bridge_pool_fraction: float,
    campaign_order: tuple[int, ...] | None = None,
    global_day_dial: bool = False,
) -> PortfolioLabelingResult:
    """Label several campaigns against one model-line-wide identifier map."""
    if campaign_order is None:
        campaign_order = tuple(range(len(campaigns)))
    n_edps, n_weeks, n_users = campaigns[0].events.shape
    labels = {
        campaign.campaign_id: np.full(campaign.events.shape, -1, dtype=np.int64)
        for campaign in campaigns
    }
    memo: dict[int, int] = {}
    bridge_size = max(101, int(round(n_users * bridge_pool_fraction)))
    for day in range(n_weeks):
        if global_day_dial:
            volumes = np.asarray(
                [campaign.events[:, day].sum() for campaign in campaigns],
                dtype=float,
            )
            if float(volumes.sum()) > 0:
                shared_dial = float(
                    np.average(
                        [dials[campaign.campaign_id][day] for campaign in campaigns],
                        weights=volumes,
                    )
                )
            else:
                shared_dial = float(np.mean([dials[campaign.campaign_id][day] for campaign in campaigns]))
        else:
            shared_dial = 0.0
        for campaign_index in campaign_order:
            campaign = campaigns[campaign_index]
            dial = shared_dial if global_day_dial else float(dials[campaign.campaign_id][day])
            for edp in range(n_edps):
                users = np.flatnonzero(campaign.events[edp, day])
                if not len(users):
                    continue
                keys, email = _identity_keys(users, edp, world.email_linkable, n_users)
                proposed = _labels_for_keys(
                    keys,
                    users,
                    edp,
                    email,
                    dial,
                    n_users,
                    bridge_size,
                )
                assigned = np.empty(len(keys), dtype=np.int64)
                for index, key in enumerate(keys.tolist()):
                    assigned[index] = memo.setdefault(int(key), int(proposed[index]))
                labels[campaign.campaign_id][edp, day, users] = assigned
    return PortfolioLabelingResult(
        method,
        labels,
        len(memo),
        "All campaigns share one ordered model-line identifier-to-VID map.",
    )


def label_campaigns_with_local_maps(
    world: SyntheticWorld,
    campaigns: list[Campaign],
    dials: dict[str, np.ndarray],
    bridge_pool_fraction: float,
) -> PortfolioLabelingResult:
    results = {
        campaign.campaign_id: label_hash_pool(
            world,
            campaign,
            dials[campaign.campaign_id],
            "campaign_local",
            bridge_pool_fraction,
            sticky=True,
        )
        for campaign in campaigns
    }
    return PortfolioLabelingResult(
        "campaign_local_maps",
        {campaign_id: result.labels for campaign_id, result in results.items()},
        sum(result.state_entries for result in results.values()),
        "Each campaign has its own map; this maximizes adaptation but fragments identities across campaigns.",
    )


def _portfolio_union(
    labels_by_campaign: dict[str, np.ndarray],
    weeks: tuple[int, ...],
    edps: tuple[int, ...],
) -> int:
    collected = []
    for labels in labels_by_campaign.values():
        selected = labels[np.ix_(edps, weeks, np.arange(labels.shape[2]))]
        collected.append(selected[selected >= 0])
    return int(len(np.unique(np.concatenate(collected)))) if collected else 0


def _portfolio_truth(
    campaigns: list[Campaign],
    weeks: tuple[int, ...],
    edps: tuple[int, ...],
) -> int:
    combined = np.zeros(campaigns[0].events.shape[2], dtype=bool)
    users = np.arange(campaigns[0].events.shape[2])
    for campaign in campaigns:
        combined |= np.any(campaign.events[np.ix_(edps, weeks, users)], axis=(0, 1))
    return int(combined.sum())


def run_portfolio_experiment(
    world: SyntheticWorld,
    campaigns: list[Campaign],
    model: DialRegressor,
    config: DailyExperimentConfig,
) -> tuple[list[dict], dict]:
    by_scenario = {campaign.scenario: campaign for campaign in campaigns}
    pair_names = (
        ("broad_awareness_control", "crm_customer_list"),
        ("website_retargeting", "app_activity_retargeting"),
        ("traffic_optimization", "lead_generation"),
        ("lookalike_prospecting", "unrelated_niche_control"),
    )
    rows: list[dict] = []
    order_differences: list[float] = []
    for left_name, right_name in pair_names:
        if left_name not in by_scenario or right_name not in by_scenario:
            continue
        pair = [by_scenario[left_name], by_scenario[right_name]]
        dials = {
            campaign.campaign_id: predict_day_dials(
                model,
                world,
                campaign,
                "cumulative",
                "full",
            )
            for campaign in pair
        }
        forward = label_campaign_portfolio(
            world,
            pair,
            dials,
            "shared_map_campaign_specific_forward",
            config.bridge_pool_fraction,
            campaign_order=(0, 1),
        )
        reverse = label_campaign_portfolio(
            world,
            pair,
            dials,
            "shared_map_campaign_specific_reverse",
            config.bridge_pool_fraction,
            campaign_order=(1, 0),
        )
        pooled = label_campaign_portfolio(
            world,
            pair,
            dials,
            "shared_map_global_daily_state",
            config.bridge_pool_fraction,
            campaign_order=(0, 1),
            global_day_dial=True,
        )
        local = label_campaigns_with_local_maps(
            world,
            pair,
            dials,
            config.bridge_pool_fraction,
        )
        methods = (forward, reverse, pooled, local)
        specs = (
            ("full_2_edps", tuple(range(config.n_weeks)), (0, 1)),
            ("weeks_5_12_5_edps", tuple(range(4, 12)), tuple(range(5))),
            ("full_5_edps", tuple(range(config.n_weeks)), tuple(range(5))),
            ("full_10_edps", tuple(range(config.n_weeks)), tuple(range(config.n_edps))),
        )
        for report_name, weeks, edps in specs:
            truth = _portfolio_truth(pair, weeks, edps)
            forward_value = _portfolio_union(forward.labels_by_campaign, weeks, edps)
            reverse_value = _portfolio_union(reverse.labels_by_campaign, weeks, edps)
            order_differences.append(abs(forward_value - reverse_value) / max(truth, 1))
            for result in methods:
                estimate = _portfolio_union(result.labels_by_campaign, weeks, edps)
                rows.append(
                    {
                        "portfolio": f"{left_name}+{right_name}",
                        "scope": "combined",
                        "method": result.method,
                        "report": report_name,
                        "truth_union": truth,
                        "estimated_union": estimate,
                        "union_relative_error": abs(estimate - truth) / max(truth, 1),
                        "signed_union_error": (estimate - truth) / max(truth, 1),
                        "state_entries": result.state_entries,
                    }
                )
            for campaign in pair:
                truth_single = _truth_union(campaign, weeks, edps)
                forward_single = _report_union(
                    forward.labels_by_campaign[campaign.campaign_id],
                    weeks,
                    edps,
                )
                reverse_single = _report_union(
                    reverse.labels_by_campaign[campaign.campaign_id],
                    weeks,
                    edps,
                )
                order_differences.append(
                    abs(forward_single - reverse_single) / max(truth_single, 1)
                )
                for result in methods:
                    estimate_single = _report_union(
                        result.labels_by_campaign[campaign.campaign_id],
                        weeks,
                        edps,
                    )
                    rows.append(
                        {
                            "portfolio": f"{left_name}+{right_name}",
                            "scope": campaign.scenario,
                            "method": result.method,
                            "report": report_name,
                            "truth_union": truth_single,
                            "estimated_union": estimate_single,
                            "union_relative_error": abs(estimate_single - truth_single)
                            / max(truth_single, 1),
                            "signed_union_error": (estimate_single - truth_single)
                            / max(truth_single, 1),
                            "state_entries": result.state_entries,
                        }
                    )
    summary = {
        "campaign_pairs": len(pair_names),
        "campaign_order_sensitivity": _summary(order_differences),
        "methods": {
            method: _summary(
                [row["union_relative_error"] for row in rows if row["method"] == method]
            )
            for method in sorted({row["method"] for row in rows})
        },
    }
    return rows, summary


def _plot_method_errors(rows: list[dict], output: Path) -> None:
    labels = {
        "fixed_model_line_pool": "Fixed model-line pool",
        "context_and_scale_only_pool": "Context + scale, stored map",
        "sticky_first_seen_pool": "Context + observed matching, stored map",
        "ordered_collision_resolved_overlap_pool": "Collision-resolved stored map",
        "fixed_marginal_overlap_atlas": "Fixed-marginal fallback allocator",
        "pair_targeted_fixed_marginal_atlas": "Pair-targeted fixed-marginal allocator",
        "full_flight_buffer_then_freeze": "Full-flight buffer",
        "oracle_online_union_oldest": "Oracle cumulative-union allocator",
        "oracle_online_venn_recent": "Oracle cumulative-Venn allocator",
        "forbidden_person_identity_oracle": "Forbidden identity oracle",
    }
    methods = [method for method in labels if any(row["method"] == method for row in rows)]
    data = [
        [100.0 * row["union_relative_error"] for row in rows if row["method"] == method]
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.boxplot(
        data,
        tick_labels=[labels[method] for method in methods],
        showfliers=False,
        vert=False,
    )
    ax.set_xlabel("Absolute union-reach error (%)")
    ax.set_title("Immutable impression labels: accuracy across report windows and EDP subsets")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_fragmentation(rows: list[dict], output: Path) -> None:
    labels = {
        "fixed_model_line_pool": "Fixed model-line pool",
        "same_day_adaptive_pool": "Same-day adaptive",
        "cumulative_adaptive_pool": "Cumulative adaptive",
        "sticky_first_seen_pool": "Stored first-seen map",
        "ordered_collision_resolved_overlap_pool": "Collision-resolved stored map",
        "fixed_marginal_overlap_atlas": "Fixed-marginal fallback allocator",
        "pair_targeted_fixed_marginal_atlas": "Pair-targeted fixed-marginal allocator",
        "full_flight_buffer_then_freeze": "Full-flight buffer",
        "oracle_online_venn_recent": "Oracle cumulative-Venn allocator",
        "forbidden_person_identity_oracle": "Forbidden identity oracle",
    }
    methods = [method for method in labels if any(row["method"] == method for row in rows)]
    fragmentation = [
        np.mean([row["stable_key_fragmentation"] for row in rows if row["method"] == method])
        for method in methods
    ]
    multiplicity = [
        np.mean([row["mean_vids_per_true_person"] for row in rows if row["method"] == method])
        for method in methods
    ]
    email_fragmentation = [
        np.mean([row["cross_edp_email_fragmentation"] for row in rows if row["method"] == method])
        for method in methods
    ]
    y = np.arange(len(methods))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    left.barh(y - 0.18, 100.0 * np.asarray(fragmentation), 0.36, label="Within-EDP ID churn")
    left.barh(y + 0.18, 100.0 * np.asarray(email_fragmentation), 0.36, label="Shared-email fragmentation")
    left.set_yticks(y, [labels[method] for method in methods])
    left.set_xlabel("Identifiers split across multiple VIDs (%)")
    left.grid(axis="x", alpha=0.25)
    left.legend(loc="lower right")
    right.barh(y, multiplicity, 0.52, color="#dd8452")
    right.set_xlabel("Mean distinct VIDs per reached true person")
    right.grid(axis="x", alpha=0.25)
    fig.suptitle("Identity fragmentation and person multiplicity")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_daily_labeling_experiment(output_dir: Path, profile: str = "quick") -> dict:
    config = DailyExperimentConfig.for_profile(profile)
    simulation = SimulationConfig(
        n_users=config.n_users,
        population_size=config.population_size,
        n_edps=config.n_edps,
        n_weeks=config.n_weeks,
        seed=config.seed,
    )
    world = make_world(simulation)
    training_roster = (
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
    )
    evaluation_roster = training_roster + (
        "linkage_shift_-1.0",
        "linkage_shift_0.0",
        "linkage_shift_1.0",
    )
    training = []
    evaluation = []
    for scenario_index, scenario in enumerate(training_roster):
        for index in range(config.training_campaigns_per_scenario):
            training.append(
                generate_campaign(
                    world,
                    scenario,
                    config.seed + 10_000 + scenario_index * 100 + index,
                    f"train_{scenario}_{index}",
                )
            )
    for scenario_index, scenario in enumerate(evaluation_roster):
        for index in range(config.evaluation_campaigns_per_scenario):
            evaluation.append(
                generate_campaign(
                    world,
                    scenario,
                    config.seed + 50_000 + scenario_index * 100 + index,
                    f"eval_{scenario}_{index}",
                )
            )
    for index, kind in enumerate(
        ("staggered_retargeting", "synchronized_retargeting", "shared_seed_then_expansion")
    ):
        evaluation.append(
            generate_temporal_stress_campaign(
                world,
                kind,
                config.seed + 90_000 + index,
                f"eval_{kind}",
            )
        )

    model, fit_records = fit_dial_regressor(world, training, config, "full")
    context_model, context_fit_records = fit_dial_regressor(
        world,
        training,
        config,
        "context_scale",
    )
    fit_records.extend(context_fit_records)
    resolved_model, resolved_fit_records = fit_collision_resolved_regressor(
        world,
        training,
        config,
    )
    fit_records.extend(resolved_fit_records)
    union_model, union_fit_records = fit_union_overlap_regressor(
        world,
        training,
        config,
    )
    fit_records.extend(union_fit_records)
    pair_capture_model = fit_pair_capture_model(world, training, config)
    static_dial = fit_static_dial(world, training, config)
    rows: list[dict] = []
    audits: dict[str, list[dict]] = {}
    dial_records: list[dict] = []
    for campaign in evaluation:
        methods = build_labeling_methods(
            model,
            context_model,
            resolved_model,
            union_model,
            pair_capture_model,
            world,
            campaign,
            config,
            static_dial,
        )
        for result in methods:
            rows.extend(evaluate_labeling(world, campaign, result))
            audits.setdefault(result.method, []).append(_consistency_audit(campaign, result))
            for day, value in enumerate(result.day_dials):
                dial_records.append(
                    {
                        "campaign_id": campaign.campaign_id,
                        "scenario": campaign.scenario,
                        "method": result.method,
                        "day": day + 1,
                        "dial": float(value),
                    }
                )

    portfolio_rows, portfolio_summary = run_portfolio_experiment(
        world,
        evaluation,
        model,
        config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "daily_labeling_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "daily_dials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dial_records[0]))
        writer.writeheader()
        writer.writerows(dial_records)
    with (output_dir / "dial_fit_records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fit_records[0]))
        writer.writeheader()
        writer.writerows(fit_records)
    with (output_dir / "cross_campaign_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(portfolio_rows[0]))
        writer.writeheader()
        writer.writerows(portfolio_rows)

    method_summary = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        method_summary[method] = {
            "union_error": _summary([row["union_relative_error"] for row in selected]),
            "signed_union_error": _summary([row["signed_union_error"] for row in selected]),
            "pair_intersection_error": _summary(
                [row["pair_intersection_error"] for row in selected]
            ),
            "marginal_reach_error": _summary(
                [row["mean_marginal_relative_error"] for row in selected]
            ),
            "population_bound_excess": _summary(
                [row["population_bound_excess"] for row in selected]
            ),
            "stable_key_fragmentation": float(
                np.mean([row["stable_key_fragmentation"] for row in selected])
            ),
            "cross_edp_email_fragmentation": float(
                np.mean([row["cross_edp_email_fragmentation"] for row in selected])
            ),
            "mean_vids_per_true_person": float(
                np.mean([row["mean_vids_per_true_person"] for row in selected])
            ),
            "mean_day_dial_churn": float(np.mean([row["day_dial_churn"] for row in selected])),
            "mean_state_entries": float(np.mean([int(row["state_entries"]) for row in selected])),
            "pool_count": int(max(int(row["pool_count"]) for row in selected)),
            "requires_ordered_days": bool(selected[0]["requires_ordered_days"]),
            "consistency": {
                "nested_report_checks": int(
                    sum(item["nested_report_checks"] for item in audits[method])
                ),
                "nested_report_violations": int(
                    sum(item["nested_report_violations"] for item in audits[method])
                ),
                "max_nested_violation": int(
                    max(item["max_nested_violation"] for item in audits[method])
                ),
                "max_replay_difference": int(
                    max(item["weeks_1_3_replay_difference"] for item in audits[method])
                ),
            },
        }
    fit_by_set = {}
    for feature_set in sorted({record["feature_set"] for record in fit_records}):
        values = [
            record["absolute_dial_error"]
            for record in fit_records
            if record["feature_set"] == feature_set
        ]
        fit_by_set[feature_set] = {
            "mean_absolute_error": float(np.mean(values)),
            "p90_absolute_error": float(np.quantile(values, 0.9)),
        }
    summary = {
        "profile": profile,
        "configuration": config.__dict__,
        "training_campaign_count": len(training),
        "evaluation_campaign_count": len(evaluation),
        "training_scenario_roster": training_roster,
        "evaluation_scenario_roster": evaluation_roster,
        "dial_fit": {
            "by_feature_set": fit_by_set,
            "feature_names": model.feature_names,
            "coefficient_count": len(model.coefficients),
            "context_only_coefficient_count": len(context_model.coefficients),
            "collision_resolved_coefficient_count": len(resolved_model.coefficients),
            "fixed_marginal_union_coefficient_count": len(union_model.coefficients),
            "pair_targeted_capture_parameter_count": pair_capture_model.parameter_count,
            "static_broad_campaign_dial": static_dial,
        },
        "methods": method_summary,
        "cross_campaign": portfolio_summary,
        "interpretation": {
            "consistency": "Every result is computed from immutable event labels, so nested reports and exact reruns are consistent by construction.",
            "fragmentation": "Day-varying stateless pools can still assign one stable input identifier to several VIDs across days, inflating multi-day reach even though reports remain logically consistent.",
            "warmup": "The buffered method avoids that fragmentation but cannot publish final labels until its warmup period ends.",
        },
    }
    (output_dir / "daily_labeling_summary.json").write_text(json.dumps(summary, indent=2))
    _plot_method_errors(rows, output_dir / "daily_labeling_error.png")
    _plot_fragmentation(rows, output_dir / "daily_labeling_fragmentation.png")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run_daily_labeling_experiment(arguments.output_dir, arguments.profile)
