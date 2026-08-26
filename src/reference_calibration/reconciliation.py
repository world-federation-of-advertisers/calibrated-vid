from __future__ import annotations

from dataclasses import dataclass, field
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class ReportKey:
    campaign_id: str
    model_name: str
    weeks: tuple[int, ...]
    edps: tuple[int, ...]
    model_line: str = "synthetic-model-line-v1"
    source_version: str = "synthetic-source-v1"
    population: str = "all-persons"

    def scope(self, n_weeks: int) -> frozenset[int]:
        return frozenset(edp * n_weeks + week for edp in self.edps for week in self.weeks)


@dataclass(frozen=True)
class ReportCandidate:
    key: ReportKey
    raw_union: float
    marginal_reaches: tuple[float, ...]
    uncertainty: float
    population_size: float


@dataclass(frozen=True)
class FinalizedReport:
    key: ReportKey
    raw_union: float
    finalized_union: float
    status: str
    lower_bound: float
    upper_bound: float
    movement: float
    slack: float
    anchor_count: int
    diagnostics: dict = field(default_factory=dict)


class ResultRegistry:
    def __init__(self, n_weeks: int, review_movement_fraction: float = 0.05):
        self.n_weeks = n_weeks
        self.review_movement_fraction = review_movement_fraction
        self._records: dict[ReportKey, FinalizedReport] = {}

    @property
    def records(self) -> tuple[FinalizedReport, ...]:
        return tuple(self._records.values())

    def finalize(self, candidate: ReportCandidate) -> FinalizedReport:
        if candidate.key in self._records:
            return self._records[candidate.key]

        comparable = {
            key.scope(self.n_weeks): record
            for key, record in self._records.items()
            if key.campaign_id == candidate.key.campaign_id
            and key.model_name == candidate.key.model_name
            and key.model_line == candidate.key.model_line
            and key.source_version == candidate.key.source_version
            and key.population == candidate.key.population
        }
        scope = candidate.key.scope(self.n_weeks)
        basic_lower = max(candidate.marginal_reaches, default=0.0)
        basic_upper = min(sum(candidate.marginal_reaches), candidate.population_size)
        desired_lower = basic_lower
        desired_upper = basic_upper
        constraints: list[tuple[str, float, str]] = []

        for old_scope, record in comparable.items():
            if old_scope <= scope:
                desired_lower = max(desired_lower, record.finalized_union)
                constraints.append(("contains_prior", record.finalized_union, str(record.key)))
            if scope <= old_scope:
                desired_upper = min(desired_upper, record.finalized_union)
                constraints.append(("contained_by_prior", record.finalized_union, str(record.key)))

        items = list(comparable.items())
        for index, (left_scope, left) in enumerate(items):
            for right_scope, right in items[index + 1 :]:
                union_scope = left_scope | right_scope
                intersection_scope = left_scope & right_scope
                intersection_value = comparable.get(intersection_scope)
                union_value = comparable.get(union_scope)
                if scope == union_scope:
                    overlap = 0.0 if not intersection_scope else (
                        intersection_value.finalized_union if intersection_value else None
                    )
                    if overlap is not None:
                        bound = left.finalized_union + right.finalized_union - overlap
                        desired_upper = min(desired_upper, bound)
                        constraints.append(("set_addition_upper", bound, f"{left.key}|{right.key}"))
                if scope == intersection_scope and union_value is not None:
                    bound = left.finalized_union + right.finalized_union - union_value.finalized_union
                    desired_upper = min(desired_upper, bound)
                    constraints.append(("set_addition_upper", bound, f"{left.key}|{right.key}"))

        for old_scope, old in items:
            union_value = comparable.get(old_scope | scope)
            intersection_scope = old_scope & scope
            intersection_value = comparable.get(intersection_scope) if intersection_scope else None
            if union_value is not None and (not intersection_scope or intersection_value is not None):
                overlap = 0.0 if not intersection_scope else intersection_value.finalized_union
                bound = union_value.finalized_union + overlap - old.finalized_union
                desired_lower = max(desired_lower, bound)
                constraints.append(("set_addition_lower", bound, str(old.key)))

        feasible = desired_lower <= desired_upper + 1e-9
        if feasible:
            finalized = min(max(candidate.raw_union, desired_lower), desired_upper)
            slack = 0.0
        else:
            scale = max(candidate.uncertainty, 1.0)

            def objective(value: float) -> float:
                movement = ((value - candidate.raw_union) / scale) ** 2
                violation = max(desired_lower - value, 0.0) ** 2 + max(value - desired_upper, 0.0) ** 2
                return movement + 1_000.0 * violation / max(candidate.population_size**2, 1.0)

            solved = minimize_scalar(objective, bounds=(basic_lower, basic_upper), method="bounded")
            finalized = float(solved.x if solved.success else min(max(candidate.raw_union, basic_lower), basic_upper))
            slack = max(desired_lower - finalized, finalized - desired_upper, 0.0)

        movement = abs(finalized - candidate.raw_union)
        movement_fraction = movement / max(candidate.raw_union, 1.0)
        status = "OK"
        if not feasible or movement_fraction > self.review_movement_fraction:
            status = "REVIEW_REQUIRED"
        elif movement > 1e-6:
            status = "RECONCILED"

        report = FinalizedReport(
            key=candidate.key,
            raw_union=float(candidate.raw_union),
            finalized_union=float(finalized),
            status=status,
            lower_bound=float(desired_lower),
            upper_bound=float(desired_upper),
            movement=float(movement),
            slack=float(slack),
            anchor_count=len(comparable),
            diagnostics={"feasible": feasible, "constraints": constraints},
        )
        self._records[candidate.key] = report
        return report

    def inject_finalized(self, report: FinalizedReport) -> None:
        """Test-only helper for constructing deliberately conflicting history."""
        self._records[report.key] = report

    def audit(self) -> dict[str, int | float]:
        by_context: dict[tuple[str, str, str, str, str], list[tuple[frozenset[int], FinalizedReport]]] = {}
        for report in self._records.values():
            context = (
                report.key.campaign_id,
                report.key.model_name,
                report.key.model_line,
                report.key.source_version,
                report.key.population,
            )
            by_context.setdefault(context, []).append((report.key.scope(self.n_weeks), report))

        monotonic_checks = monotonic_violations = 0
        set_checks = set_violations = 0
        max_violation = 0.0
        for records in by_context.values():
            lookup = {scope: report for scope, report in records}
            for left_scope, left in records:
                for right_scope, right in records:
                    if left_scope < right_scope:
                        monotonic_checks += 1
                        violation = left.finalized_union - right.finalized_union
                        if violation > 1e-6:
                            monotonic_violations += 1
                            max_violation = max(max_violation, violation)
            for index, (left_scope, left) in enumerate(records):
                for right_scope, right in records[index + 1 :]:
                    union = lookup.get(left_scope | right_scope)
                    intersection_scope = left_scope & right_scope
                    intersection = lookup.get(intersection_scope) if intersection_scope else None
                    if union is None or (intersection_scope and intersection is None):
                        continue
                    set_checks += 1
                    intersection_value = 0.0 if not intersection_scope else intersection.finalized_union
                    violation = union.finalized_union + intersection_value - left.finalized_union - right.finalized_union
                    if violation > 1e-6:
                        set_violations += 1
                        max_violation = max(max_violation, violation)
        return {
            "monotonic_checks": monotonic_checks,
            "monotonic_violations": monotonic_violations,
            "set_arithmetic_checks": set_checks,
            "set_arithmetic_violations": set_violations,
            "max_violation": max_violation,
        }
