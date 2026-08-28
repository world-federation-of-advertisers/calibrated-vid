from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .population import Campaign, SyntheticWorld
from .sets import (
    exact_cells_from_membership,
    inclusive_intersections,
    members,
    project_to_bounded_sum,
    union_values,
)


@dataclass(frozen=True)
class ReportObservation:
    campaign_id: str
    scenario: str
    weeks: tuple[int, ...]
    edps: tuple[int, ...]
    person_weight: float
    global_masks: np.ndarray
    reach_fractions: np.ndarray
    truth_exact_cells: np.ndarray
    truth_intersections: np.ndarray
    truth_unions: np.ndarray
    baseline_intersections: np.ndarray
    baseline_unions: np.ndarray
    email_intersections: np.ndarray
    reference_intersections: np.ndarray
    collision_floor: np.ndarray
    reference_signal: np.ndarray
    objectives: tuple[str, ...]
    audience_strategies: tuple[str, ...]
    demographic_labels: tuple[str, ...]
    demographic_population: np.ndarray
    edp_demographic_reaches: np.ndarray
    truth_demographic_union: np.ndarray
    baseline_demographic_union: np.ndarray


@dataclass(frozen=True)
class CalibrationDataset:
    campaign_ids: np.ndarray
    subset_masks: np.ndarray
    subset_orders: np.ndarray
    log_scale: np.ndarray
    k0: np.ndarray
    signal: np.ndarray
    truth: np.ndarray
    weight: np.ndarray

    def select_campaigns(self, campaign_ids: set[str]) -> "CalibrationDataset":
        selected = np.array([value in campaign_ids for value in self.campaign_ids], dtype=bool)
        return CalibrationDataset(
            campaign_ids=self.campaign_ids[selected],
            subset_masks=self.subset_masks[selected],
            subset_orders=self.subset_orders[selected],
            log_scale=self.log_scale[selected],
            k0=self.k0[selected],
            signal=self.signal[selected],
            truth=self.truth[selected],
            weight=self.weight[selected],
        )


def _collision_rng(campaign_id: str, weeks: tuple[int, ...], global_mask: int, seed: int):
    """Return a report-invariant RNG for one measured EDP intersection.

    The same campaign, time window, and EDP subset must produce the same
    Reference-ID collision contribution whether the subset is requested alone
    or as part of a larger report.
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update(f"{seed}|{campaign_id}|{weeks}|{global_mask}".encode())
    return np.random.default_rng(int.from_bytes(digest.digest(), "little"))


def _global_masks(edps: tuple[int, ...]) -> np.ndarray:
    result = np.zeros(1 << len(edps), dtype=np.int64)
    for local_mask in range(1, 1 << len(edps)):
        global_mask = 0
        for local_index, edp in enumerate(edps):
            if local_mask & (1 << local_index):
                global_mask |= 1 << edp
        result[local_mask] = global_mask
    return result


def measure_report(
    world: SyntheticWorld,
    campaign: Campaign,
    weeks: tuple[int, ...],
    edps: tuple[int, ...],
) -> ReportObservation:
    config = world.config
    reached = np.any(campaign.events[np.ix_(edps, weeks, np.arange(config.n_users))], axis=1)
    exact = exact_cells_from_membership(reached, config.person_weight)
    truth_intersections = inclusive_intersections(exact)
    truth_unions = union_values(exact)
    marginals = np.array([truth_intersections[1 << i] for i in range(len(edps))])
    reach_fractions = marginals / config.population_size

    size = 1 << len(edps)
    global_masks = _global_masks(edps)
    baseline_intersections = np.zeros(size, dtype=float)
    baseline_unions = np.zeros(size, dtype=float)
    for subset in range(1, size):
        local = members(subset, len(edps))
        fractions = reach_fractions[list(local)]
        if len(local) == 1:
            baseline_intersections[subset] = marginals[local[0]]
        else:
            baseline_intersections[subset] = config.population_size * float(np.prod(fractions))
        baseline_unions[subset] = config.population_size * (
            1.0 - float(np.prod(1.0 - fractions))
        )

    # The demographic-agnostic VID labeler receives email and proprietary IDs
    # as separate inputs.  When the same email is present at several EDPs, the
    # email-derived VID is a direct cross-EDP identity anchor.  These aggregate
    # counts model the resulting VID overlap; they are not the downstream
    # Reference-ID measurement and contain no 5B-pool collision contribution.
    visible_membership = reached & world.email_linkable[np.array(edps)]
    visible_exact = exact_cells_from_membership(visible_membership, config.person_weight)
    visible_intersections = inclusive_intersections(visible_exact)
    reference = np.zeros(size, dtype=float)
    collision_floor = np.zeros(size, dtype=float)
    for subset in range(1, size):
        local = members(subset, len(edps))
        if len(local) == 1:
            reference[subset] = marginals[local[0]]
            continue
        occupied = -np.expm1(-marginals[list(local)] / config.reference_pool_size)
        floor = config.reference_pool_size * float(np.prod(occupied))
        collision_floor[subset] = floor
        global_mask = int(global_masks[subset])
        rng = _collision_rng(campaign.campaign_id, weeks, global_mask, config.seed)
        collision = float(rng.poisson(max(floor, 0.0)))
        reference[subset] = visible_intersections[subset] + collision

    demographic_count = len(world.demographic_labels)
    any_reached = np.any(reached, axis=0)
    truth_demographic_union = (
        np.bincount(
            world.true_demographic[any_reached],
            minlength=demographic_count,
        ).astype(float)
        * config.person_weight
    )
    edp_demographic_reaches = np.zeros((len(edps), demographic_count), dtype=float)
    for local_index in range(len(edps)):
        edp_demographic_reaches[local_index] = (
            np.bincount(
                world.vid_demographic[reached[local_index]],
                minlength=demographic_count,
            ).astype(float)
            * config.person_weight
        )
    raw_demographic_union = np.zeros(demographic_count, dtype=float)
    for demographic in range(demographic_count):
        population = max(float(world.vid_demographic_population[demographic]), 1.0)
        fractions = np.clip(edp_demographic_reaches[:, demographic] / population, 0.0, 1.0)
        raw_demographic_union[demographic] = population * (
            1.0 - float(np.prod(1.0 - fractions))
        )
    baseline_total = float(baseline_unions[-1])
    if raw_demographic_union.sum() <= 0:
        raw_demographic_union = world.vid_demographic_population.copy()
    scaled_demographic_union = (
        raw_demographic_union * baseline_total / float(raw_demographic_union.sum())
    )
    baseline_demographic_union = project_to_bounded_sum(
        scaled_demographic_union,
        baseline_total,
        upper=world.true_demographic_population,
    )

    return ReportObservation(
        campaign_id=campaign.campaign_id,
        scenario=campaign.scenario,
        weeks=tuple(weeks),
        edps=tuple(edps),
        person_weight=config.person_weight,
        global_masks=global_masks,
        reach_fractions=reach_fractions,
        truth_exact_cells=exact,
        truth_intersections=truth_intersections,
        truth_unions=truth_unions,
        baseline_intersections=baseline_intersections,
        baseline_unions=baseline_unions,
        email_intersections=visible_intersections,
        reference_intersections=reference,
        collision_floor=collision_floor,
        reference_signal=reference - collision_floor,
        objectives=tuple(campaign.objectives[edp] for edp in edps),
        audience_strategies=tuple(campaign.audience_strategies[edp] for edp in edps),
        demographic_labels=world.demographic_labels,
        demographic_population=world.true_demographic_population.copy(),
        edp_demographic_reaches=edp_demographic_reaches,
        truth_demographic_union=truth_demographic_union,
        baseline_demographic_union=baseline_demographic_union,
    )


def calibration_dataset(
    observations: list[ReportObservation],
    minimum_intersection: float,
) -> CalibrationDataset:
    campaign_ids: list[str] = []
    subset_masks: list[int] = []
    subset_orders: list[int] = []
    log_scale: list[float] = []
    k0: list[float] = []
    signal: list[float] = []
    truth: list[float] = []
    weight: list[float] = []

    for observation in observations:
        for local_mask in range(1, len(observation.global_masks)):
            order = local_mask.bit_count()
            if order < 2:
                continue
            reference = observation.baseline_intersections[local_mask]
            visible = observation.reference_signal[local_mask]
            if reference < minimum_intersection or visible <= 0:
                continue
            local_members = members(local_mask, len(observation.edps))
            geometric_mean = float(
                np.exp(np.mean(np.log(np.maximum(observation.reach_fractions[list(local_members)], 1e-9))))
            )
            campaign_ids.append(observation.campaign_id)
            subset_masks.append(int(observation.global_masks[local_mask]))
            subset_orders.append(order)
            log_scale.append(float(np.log(max(geometric_mean, 1e-9))))
            k0.append(reference)
            signal.append(visible)
            truth.append(observation.truth_intersections[local_mask])
            weight.append(float(np.sqrt(max(reference, 1.0))))

    return CalibrationDataset(
        campaign_ids=np.asarray(campaign_ids, dtype=object),
        subset_masks=np.asarray(subset_masks, dtype=np.int64),
        subset_orders=np.asarray(subset_orders, dtype=np.int8),
        log_scale=np.asarray(log_scale, dtype=float),
        k0=np.asarray(k0, dtype=float),
        signal=np.asarray(signal, dtype=float),
        truth=np.asarray(truth, dtype=float),
        weight=np.asarray(weight, dtype=float),
    )
