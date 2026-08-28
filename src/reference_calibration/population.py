from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


SCENARIOS = (
    "representative",
    "small_vs_large_nonreach",
    "two_small_correlated",
    "two_small_disjoint",
    "all_small_correlated",
    "mixed_objectives",
    "high_matchability_remarketing",
    "low_matchability_targeting",
)

# Product-facing scenarios used by the explanatory notebook.  These are
# stylized audience-selection mechanisms, not estimates of Meta production
# distributions.  SCENARIOS above remains the stable validation-suite roster.
META_CAMPAIGN_SCENARIOS = (
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

# Plain-language assumptions shown in the scenario notebook.  The qualitative
# levels describe the synthetic generator; they are not estimates of Meta's
# production delivery or match rates.
META_SCENARIO_DESCRIPTIONS = {
    "broad_awareness_control": {
        "objective": "Awareness",
        "audience": "Broad awareness delivery with few audience restrictions",
        "chart_label": "Broad awareness",
        "volume": "Large",
        "cross_edp_similarity": "Low to moderate: each EDP reaches a broad but not identical audience",
        "reference_matchability": "Typical for the general population",
        "intuition": "Control case: broad delivery is closest to the campaigns on which a conventional reach model is expected to work well.",
    },
    "traffic_optimization": {
        "objective": "Traffic",
        "audience": "Interest targeting with click or landing-page optimization",
        "chart_label": "Traffic optimization",
        "volume": "Medium",
        "cross_edp_similarity": "Moderate: optimization can favor similar click-prone people",
        "reference_matchability": "Typical; clicking does not require a shared email",
        "intuition": "Tests a campaign that is more selected than awareness but is not a direct retargeting audience.",
    },
    "video_engagement_retargeting": {
        "objective": "Engagement",
        "audience": "Engagement Custom Audience of prior video viewers or engagers",
        "chart_label": "Engagement retargeting",
        "volume": "Small to medium",
        "cross_edp_similarity": "High: the same warm audience can be selected repeatedly",
        "reference_matchability": "Slightly below typical; engagement does not require email",
        "intuition": "Tests high true duplication without assuming that the duplicated people are especially easy to match by email.",
    },
    "lead_generation": {
        "objective": "Leads",
        "audience": "Instant Form, call, message, or signup optimization",
        "chart_label": "Lead generation",
        "volume": "Small",
        "cross_edp_similarity": "High: EDPs can converge on the same high-intent people",
        "reference_matchability": "Above typical because lead actions often collect contact details",
        "intuition": "Tests a small conversion-oriented audience with both elevated duplication and elevated email visibility.",
    },
    "sales_prospecting": {
        "objective": "Sales",
        "audience": "Prospecting optimized toward likely purchasers",
        "chart_label": "Sales prospecting",
        "volume": "Small to medium",
        "cross_edp_similarity": "High: optimization can converge on a shared pool of likely buyers",
        "reference_matchability": "Moderately above typical",
        "intuition": "Tests algorithmic convergence on likely buyers without using a direct first-party retargeting list.",
    },
    "website_retargeting": {
        "objective": "Sales",
        "audience": "Website Custom Audience of recent visitors or cart viewers",
        "chart_label": "Website retargeting",
        "volume": "Small",
        "cross_edp_similarity": "Very high: campaigns start from a similar first-party visitor pool",
        "reference_matchability": "Above typical; some visitors log in, buy, or otherwise provide email",
        "intuition": "Tests the classic small retargeting case in which true cross-EDP overlap is much higher than a population-rate model expects.",
    },
    "crm_customer_list": {
        "objective": "Sales",
        "audience": "Customer-list Custom Audience uploaded from CRM records",
        "chart_label": "Customer list",
        "volume": "Very small",
        "cross_edp_similarity": "Very high: each EDP receives substantially the same seed list",
        "reference_matchability": "Very high because the source audience is already email-rich",
        "intuition": "Tests the strongest upward matchability shift: a tiny audience selected from records that commonly include email.",
    },
    "catalog_retargeting": {
        "objective": "Sales",
        "audience": "Catalog retargeting or dynamic product ads",
        "chart_label": "Catalog retargeting",
        "volume": "Small",
        "cross_edp_similarity": "High, but some EDP pairs share more products and users than others",
        "reference_matchability": "Above typical",
        "intuition": "Tests stable pair-specific behavior rather than assuming every EDP combination matches equally well.",
    },
    "lookalike_prospecting": {
        "objective": "Sales",
        "audience": "Lookalike audience built from a customer seed",
        "chart_label": "Lookalike prospecting",
        "volume": "Medium",
        "cross_edp_similarity": "Moderate to high: the same seed influences delivery, but the audience expands",
        "reference_matchability": "Slightly above typical",
        "intuition": "Tests a middle ground between direct retargeting and broad prospecting.",
    },
    "advantage_audience_expansion": {
        "objective": "Sales",
        "audience": "Advantage+ audience expansion beyond the starting audience",
        "chart_label": "Audience expansion",
        "volume": "Medium to large",
        "cross_edp_similarity": "Moderate: the starting signal is shared, but delivery broadens",
        "reference_matchability": "Near typical after expansion",
        "intuition": "Tests whether a method handles a campaign that begins with a selected audience but expands toward broad delivery.",
    },
    "app_activity_retargeting": {
        "objective": "App promotion",
        "audience": "App Activity Custom Audience of installers or in-app users",
        "chart_label": "App retargeting",
        "volume": "Small",
        "cross_edp_similarity": "High: the same app users can be retargeted repeatedly",
        "reference_matchability": "Below typical when app identity is stronger than email identity",
        "intuition": "Tests high true duplication with a downward email-matchability shift—the opposite of a customer-list campaign.",
    },
    "unrelated_niche_control": {
        "objective": "Traffic",
        "audience": "Different niche interests or exclusions at each EDP",
        "chart_label": "Unrelated niches",
        "volume": "Small",
        "cross_edp_similarity": "Low: the EDP audiences are intentionally different",
        "reference_matchability": "Typical among the people who are reached",
        "intuition": "Negative control: the calibration should not invent duplicate reach merely because the campaigns are small.",
    },
    "mixed_funnel_portfolio": {
        "objective": "Mixed",
        "audience": "One report combining broad, traffic, lead, and retargeting campaigns",
        "chart_label": "Mixed-funnel report",
        "volume": "Mixed",
        "cross_edp_similarity": "Mixed by EDP",
        "reference_matchability": "Mixed by EDP and audience source",
        "intuition": "Tests the realistic case in which a cross-media report combines campaigns with different objectives and audience construction.",
    },
}

AGE_GROUPS = ("18-34", "35-54", "55+")
GENDER_GROUPS = ("female", "male")
GEO_GROUPS = ("northeast", "south_central", "west")
DEMOGRAPHIC_LABELS = tuple(
    f"{age} | {gender} | {geo}"
    for age in AGE_GROUPS
    for gender in GENDER_GROUPS
    for geo in GEO_GROUPS
)

CAMPAIGN_OBJECTIVES = (
    "awareness",
    "traffic",
    "engagement",
    "leads",
    "app_promotion",
    "sales",
)

AUDIENCE_STRATEGIES = (
    "broad",
    "interest",
    "engagement_retargeting",
    "lead_optimization",
    "sales_prospecting",
    "website_retargeting",
    "customer_list",
    "catalog_retargeting",
    "lookalike",
    "audience_expansion",
    "app_retargeting",
    "niche",
)


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value)))


def _intercept_for_mean(linear_score: np.ndarray, target: float) -> float:
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if float(_sigmoid(linear_score + mid).mean()) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass(frozen=True)
class SyntheticWorld:
    config: SimulationConfig
    broad_factor: np.ndarray
    segments: np.ndarray
    matchability: np.ndarray
    email_linkable: np.ndarray
    email_coverage: np.ndarray
    email_agreement: np.ndarray
    target_link_probability: np.ndarray
    realized_link_probability: np.ndarray
    true_demographic: np.ndarray
    vid_demographic: np.ndarray
    demographic_labels: tuple[str, ...]
    true_demographic_population: np.ndarray
    vid_demographic_population: np.ndarray


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    scenario: str
    events: np.ndarray
    final_reach_fraction: np.ndarray
    objectives: tuple[str, ...]
    audience_strategies: tuple[str, ...]


def _noisy_category_labels(
    truth: np.ndarray,
    n_categories: int,
    base_accuracy: float,
    matchability: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a stable, imperfect demographic label for the VID model."""
    accuracy = np.clip(base_accuracy + 0.06 * np.tanh(matchability), 0.55, 0.99)
    result = np.asarray(truth, dtype=np.int16).copy()
    changed = rng.random(len(result)) > accuracy
    alternatives = rng.integers(0, n_categories - 1, size=int(changed.sum()))
    original = result[changed]
    alternatives += alternatives >= original
    result[changed] = alternatives
    return result


def _campaign_context(scenario: str, n_edps: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return per-EDP objective and audience-strategy labels."""
    fixed = {
        "representative": ("awareness", "broad"),
        "two_small_correlated": ("sales", "website_retargeting"),
        "two_small_disjoint": ("traffic", "niche"),
        "all_small_correlated": ("engagement", "engagement_retargeting"),
        "high_matchability_remarketing": ("sales", "customer_list"),
        "low_matchability_targeting": ("app_promotion", "app_retargeting"),
        "broad_awareness_control": ("awareness", "broad"),
        "traffic_optimization": ("traffic", "interest"),
        "video_engagement_retargeting": ("engagement", "engagement_retargeting"),
        "lead_generation": ("leads", "lead_optimization"),
        "sales_prospecting": ("sales", "sales_prospecting"),
        "website_retargeting": ("sales", "website_retargeting"),
        "crm_customer_list": ("sales", "customer_list"),
        "catalog_retargeting": ("sales", "catalog_retargeting"),
        "lookalike_prospecting": ("sales", "lookalike"),
        "advantage_audience_expansion": ("sales", "audience_expansion"),
        "app_activity_retargeting": ("app_promotion", "app_retargeting"),
        "unrelated_niche_control": ("traffic", "niche"),
    }
    if scenario in ("mixed_objectives", "mixed_funnel_portfolio"):
        cycle = (
            ("awareness", "broad"),
            ("traffic", "interest"),
            ("leads", "lead_optimization"),
            ("sales", "website_retargeting"),
        )
        pairs = tuple(cycle[index % len(cycle)] for index in range(n_edps))
    elif scenario.startswith("linkage_shift_"):
        pairs = tuple(("sales", "sales_prospecting") for _ in range(n_edps))
    elif scenario == "small_vs_large_nonreach":
        pairs = tuple(
            ("awareness", "broad")
            if index % 4 == 0
            else ("sales", "website_retargeting")
            if index % 4 == 1
            else ("traffic", "interest")
            if index % 4 == 2
            else ("leads", "lead_optimization")
            for index in range(n_edps)
        )
    else:
        objective, strategy = fixed[scenario]
        pairs = tuple((objective, strategy) for _ in range(n_edps))
    return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)


def make_world(config: SimulationConfig) -> SyntheticWorld:
    rng = np.random.default_rng(config.seed)
    broad = rng.normal(size=config.n_users)
    segments = rng.normal(size=(5, config.n_users))
    matchability = rng.normal(size=config.n_users)

    age_score = 0.90 * segments[0] + 0.35 * rng.normal(size=config.n_users)
    age_thresholds = np.quantile(age_score, (0.42, 0.76))
    true_age = np.digitize(age_score, age_thresholds).astype(np.int16)
    true_gender = (
        rng.random(config.n_users) < _sigmoid(0.18 * segments[1])
    ).astype(np.int16)
    geo_score = 0.75 * segments[2] + 0.55 * rng.normal(size=config.n_users)
    geo_thresholds = np.quantile(geo_score, (0.34, 0.72))
    true_geo = np.digitize(geo_score, geo_thresholds).astype(np.int16)

    vid_age = _noisy_category_labels(true_age, len(AGE_GROUPS), 0.84, matchability, rng)
    vid_gender = _noisy_category_labels(
        true_gender,
        len(GENDER_GROUPS),
        0.94,
        matchability,
        rng,
    )
    vid_geo = _noisy_category_labels(true_geo, len(GEO_GROUPS), 0.88, matchability, rng)
    true_demographic = (
        (true_age * len(GENDER_GROUPS) + true_gender) * len(GEO_GROUPS) + true_geo
    ).astype(np.int16)
    vid_demographic = (
        (vid_age * len(GENDER_GROUPS) + vid_gender) * len(GEO_GROUPS) + vid_geo
    ).astype(np.int16)
    true_demographic_population = (
        np.bincount(true_demographic, minlength=len(DEMOGRAPHIC_LABELS)).astype(float)
        * config.person_weight
    )
    vid_demographic_population = (
        np.bincount(vid_demographic, minlength=len(DEMOGRAPHIC_LABELS)).astype(float)
        * config.person_weight
    )

    coverage = np.linspace(0.95, 0.10, config.n_edps)
    agreement = np.linspace(0.72, 0.52, config.n_edps)
    target_link = coverage * agreement

    # A global factor makes email use correlated across EDPs. Cluster factors
    # make selected EDP pairs match better than their marginal rates imply.
    cluster_count = max(1, (config.n_edps + 1) // 2)
    cluster_factors = rng.normal(size=(cluster_count, config.n_users))
    edp_noise = rng.normal(size=(config.n_edps, config.n_users))
    email_linkable = np.zeros((config.n_edps, config.n_users), dtype=bool)
    for edp in range(config.n_edps):
        cluster = min(edp // 2, cluster_count - 1)
        linear = (
            0.80 * matchability
            + 0.70 * cluster_factors[cluster]
            + 0.25 * edp_noise[edp]
        )
        intercept = _intercept_for_mean(linear, float(target_link[edp]))
        probability = _sigmoid(linear + intercept)
        email_linkable[edp] = rng.random(config.n_users) < probability

    return SyntheticWorld(
        config=config,
        broad_factor=broad,
        segments=segments,
        matchability=matchability,
        email_linkable=email_linkable,
        email_coverage=coverage,
        email_agreement=agreement,
        target_link_probability=target_link,
        realized_link_probability=email_linkable.mean(axis=1),
        true_demographic=true_demographic,
        vid_demographic=vid_demographic,
        demographic_labels=DEMOGRAPHIC_LABELS,
        true_demographic_population=true_demographic_population,
        vid_demographic_population=vid_demographic_population,
    )


def _scenario_parameters(world: SyntheticWorld, scenario: str, rng: np.random.Generator):
    n = world.config.n_edps
    independent = rng.normal(size=(n, world.config.n_users))
    common = world.broad_factor
    segment0, segment1 = world.segments[0], world.segments[1]
    match = world.matchability

    if scenario == "representative":
        reach = rng.uniform(0.45, 0.78, size=n)
        scores = 0.15 * common[None, :] + 0.10 * match[None, :] + 0.98 * independent
        timing_match_bias = 0.30
    elif scenario == "small_vs_large_nonreach":
        reach = np.linspace(0.35, 0.12, n)
        reach[1] = 0.055
        reach[3] = 0.035
        scores = 0.20 * common[None, :] + 0.98 * independent
        scores[0] = 0.85 * segment0 + 0.45 * independent[0]
        scores[1] = 0.95 * segment0 + 0.35 * match + 0.30 * independent[1]
        scores[3] = 0.80 * segment0 + 0.25 * match + 0.45 * independent[3]
        timing_match_bias = 0.15
    elif scenario == "two_small_correlated":
        reach = np.full(n, 0.012)
        reach[:2] = (0.060, 0.045)
        scores = independent
        scores[0] = 0.95 * segment0 + 0.25 * match + 0.25 * independent[0]
        scores[1] = 0.95 * segment0 + 0.25 * match + 0.25 * independent[1]
        timing_match_bias = 0.10
    elif scenario == "two_small_disjoint":
        reach = np.full(n, 0.012)
        reach[:2] = (0.060, 0.045)
        scores = independent
        scores[0] = 0.95 * segment0 + 0.20 * match + 0.25 * independent[0]
        scores[1] = -0.95 * segment0 + 0.20 * match + 0.25 * independent[1]
        timing_match_bias = 0.10
    elif scenario == "all_small_correlated":
        reach = np.linspace(0.080, 0.025, n)
        scores = 0.88 * segment0[None, :] + 0.20 * match[None, :] + 0.35 * independent
        timing_match_bias = 0.10
    elif scenario == "mixed_objectives":
        reach = np.array(
            [0.40 if i % 3 == 0 else 0.055 if i % 3 == 1 else 0.16 for i in range(n)],
            dtype=float,
        )
        scores = 0.35 * common[None, :] + 0.40 * segment0[None, :] + 0.75 * independent
        for edp in range(1, n, 3):
            scores[edp] = 0.80 * segment1 + 0.35 * match + 0.35 * independent[edp]
        timing_match_bias = 0.15
    elif scenario == "high_matchability_remarketing":
        reach = np.linspace(0.11, 0.025, n)
        scores = 0.92 * match[None, :] + 0.30 * segment0[None, :] + 0.25 * independent
        timing_match_bias = 0.35
    elif scenario == "low_matchability_targeting":
        reach = np.linspace(0.11, 0.025, n)
        scores = -0.92 * match[None, :] + 0.30 * segment1[None, :] + 0.25 * independent
        timing_match_bias = -0.20
    elif scenario == "broad_awareness_control":
        # Broad delivery is deliberately close to the campaigns used for
        # calibration and is therefore a negative control for over-correction.
        reach = rng.uniform(0.45, 0.78, size=n)
        scores = 0.15 * common[None, :] + 0.10 * match[None, :] + 0.98 * independent
        timing_match_bias = 0.30
    elif scenario == "traffic_optimization":
        # Link-click or landing-page-view optimization selects a somewhat more
        # similar audience across EDPs than broad reach, but is still sizable.
        reach = np.linspace(0.30, 0.08, n)
        scores = (
            0.30 * common[None, :]
            + 0.25 * segment0[None, :]
            + 0.05 * match[None, :]
            + 0.90 * independent
        )
        timing_match_bias = 0.12
    elif scenario == "video_engagement_retargeting":
        # A shared pool of prior video viewers or engagers creates a warm,
        # correlated audience without assuming unusually high email linkage.
        reach = np.linspace(0.16, 0.035, n)
        scores = (
            0.70 * segment0[None, :]
            + 0.20 * common[None, :]
            - 0.08 * match[None, :]
            + 0.55 * independent
        )
        timing_match_bias = 0.05
    elif scenario == "lead_generation":
        # Lead optimization concentrates delivery on a common high-intent
        # segment with moderately elevated contactability.
        reach = np.linspace(0.11, 0.025, n)
        scores = (
            0.60 * segment1[None, :]
            + 0.30 * match[None, :]
            + 0.55 * independent
        )
        timing_match_bias = 0.18
    elif scenario == "sales_prospecting":
        # Purchase optimization is narrower than broad reach and can make
        # delivery across EDPs converge on the same likely buyers.
        reach = np.linspace(0.18, 0.035, n)
        scores = (
            0.55 * segment0[None, :]
            + 0.35 * segment1[None, :]
            + 0.25 * match[None, :]
            + 0.60 * independent
        )
        timing_match_bias = 0.15
    elif scenario == "website_retargeting":
        # Website visitors and cart viewers are a small shared seed, producing
        # substantially more cross-EDP overlap than a population-rate model.
        reach = np.linspace(0.08, 0.018, n)
        scores = (
            0.95 * segment0[None, :]
            + 0.25 * match[None, :]
            + 0.30 * independent
        )
        timing_match_bias = 0.15
    elif scenario == "crm_customer_list":
        # A customer-list Custom Audience is intentionally both narrow and
        # enriched for people with a usable common email.  The same email can
        # anchor the agnostic VID and can separately generate matching
        # calibration Reference IDs.
        reach = np.linspace(0.065, 0.012, n)
        scores = (
            1.10 * match[None, :]
            + 0.45 * segment0[None, :]
            + 0.20 * independent
        )
        timing_match_bias = 0.35
    elif scenario == "catalog_retargeting":
        # Dynamic catalog campaigns share commerce intent, while alternating
        # product groups make some EDP pairs more similar than others.
        reach = np.linspace(0.09, 0.015, n)
        scores = np.zeros_like(independent)
        for edp in range(n):
            product_segment = segment0 if (edp // 2) % 2 == 0 else segment1
            scores[edp] = (
                0.85 * product_segment
                + 0.20 * match
                + 0.35 * independent[edp]
            )
        timing_match_bias = 0.14
    elif scenario == "lookalike_prospecting":
        # A lookalike is broader than its seed but keeps a shared latent
        # resemblance, yielding moderate rather than extreme overlap.
        reach = np.linspace(0.28, 0.07, n)
        scores = (
            0.55 * segment0[None, :]
            + 0.25 * common[None, :]
            + 0.10 * match[None, :]
            + 0.75 * independent
        )
        timing_match_bias = 0.12
    elif scenario == "advantage_audience_expansion":
        # Audience expansion blends a useful seed signal with broad delivery.
        # It should sit between a narrow Custom Audience and broad prospecting.
        reach = np.linspace(0.35, 0.09, n)
        scores = (
            0.45 * segment0[None, :]
            + 0.35 * common[None, :]
            + 0.15 * match[None, :]
            + 0.75 * independent
        )
        timing_match_bias = 0.12
    elif scenario == "app_activity_retargeting":
        # App installers or in-app-event users can be strongly correlated while
        # being less email-matchable than customer-list audiences.
        reach = np.linspace(0.10, 0.018, n)
        scores = (
            0.85 * segment1[None, :]
            - 0.25 * match[None, :]
            + 0.35 * independent
        )
        timing_match_bias = -0.12
    elif scenario == "unrelated_niche_control":
        # Different interest niches or explicit exclusions create little true
        # overlap.  A calibration method must not invent duplicate reach.
        reach = np.linspace(0.09, 0.02, n)
        scores = np.zeros_like(independent)
        for edp in range(n):
            direction = 1.0 if edp % 2 == 0 else -1.0
            segment = segment0 if edp % 4 < 2 else segment1
            scores[edp] = direction * 0.95 * segment + 0.25 * independent[edp]
        timing_match_bias = 0.05
    elif scenario == "mixed_funnel_portfolio":
        # A single report can combine broad awareness, traffic-sized delivery,
        # and narrow lead/sales-like campaigns from different EDPs.
        reach = np.array(
            [0.42 if i % 4 == 0 else 0.20 if i % 4 == 1 else 0.075 if i % 4 == 2 else 0.035 for i in range(n)],
            dtype=float,
        )
        scores = np.zeros_like(independent)
        for edp in range(n):
            if edp % 4 == 0:
                scores[edp] = 0.20 * common + 0.95 * independent[edp]
            elif edp % 4 == 1:
                scores[edp] = 0.35 * common + 0.25 * segment0 + 0.80 * independent[edp]
            elif edp % 4 == 2:
                scores[edp] = 0.65 * segment1 + 0.25 * match + 0.45 * independent[edp]
            else:
                scores[edp] = 0.90 * segment0 + 0.35 * match + 0.30 * independent[edp]
        timing_match_bias = 0.15
    elif scenario.startswith("linkage_shift_"):
        shift = float(scenario.removeprefix("linkage_shift_"))
        reach = np.linspace(0.18, 0.025, n)
        independent_weight = max(0.30, 1.0 - 0.55 * abs(shift))
        scores = (
            shift * match[None, :]
            + 0.45 * segment0[None, :]
            + independent_weight * independent
        )
        timing_match_bias = 0.15 * shift
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    return np.clip(reach, 0.001, 0.95), scores, timing_match_bias


def _bursty_schedule(n_weeks: int, rng: np.random.Generator) -> np.ndarray:
    weeks = np.arange(n_weeks, dtype=float)
    peak1, peak2 = rng.choice(n_weeks, size=2, replace=False)
    schedule = (
        0.08
        + np.exp(-0.5 * ((weeks - peak1) / rng.uniform(0.7, 1.8)) ** 2)
        + rng.uniform(0.35, 1.0)
        * np.exp(-0.5 * ((weeks - peak2) / rng.uniform(0.8, 2.4)) ** 2)
    )
    schedule *= rng.uniform(0.75, 1.25, size=n_weeks)
    return schedule / schedule.sum()


def generate_campaign(
    world: SyntheticWorld,
    scenario: str,
    seed: int,
    campaign_id: str,
    similarity_multiplier: float = 1.0,
    matchability_shift: float = 0.0,
) -> Campaign:
    rng = np.random.default_rng(seed)
    reach, scores, timing_match_bias = _scenario_parameters(world, scenario, rng)
    if similarity_multiplier <= 0:
        raise ValueError("similarity_multiplier must be positive")

    # Separate the part of the ranking signal shared across EDPs from each
    # EDP's deviation.  Scaling only the shared component supplies a simple
    # sensitivity axis: values below one weaken cross-EDP audience similarity;
    # values above one strengthen it without changing the requested reach.
    shared_score = np.mean(scores, axis=0, keepdims=True)
    scores = similarity_multiplier * shared_score + (scores - shared_score)

    # A positive shift makes the selected audience more likely than usual to
    # supply the same usable email across EDPs; a negative shift makes it less
    # likely.  This affects both email-anchored VID behavior and the separate
    # calibration signal, while the base scenario's selection remains intact.
    scores = scores + matchability_shift * world.matchability[None, :]

    n, weeks, users = world.config.n_edps, world.config.n_weeks, world.config.n_users
    objectives, audience_strategies = _campaign_context(scenario, n)
    events = np.zeros((n, weeks, users), dtype=bool)

    for edp in range(n):
        audience_size = max(1, int(round(reach[edp] * users)))
        selected = np.argpartition(scores[edp], -audience_size)[-audience_size:]
        base = _bursty_schedule(weeks, rng)
        trend = np.linspace(1.0, -1.0, weeks)
        adjusted = np.log(np.maximum(base, 1e-12))[None, :] + (
            timing_match_bias * world.matchability[selected, None] * trend[None, :]
        )
        adjusted -= adjusted.max(axis=1, keepdims=True)
        probability = np.exp(adjusted)
        probability /= probability.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(probability, axis=1)
        first = (rng.random(len(selected))[:, None] > cumulative).sum(axis=1)
        events[edp, first, selected] = True

        repeat_probability = np.clip(0.035 + 0.18 * base, 0.0, 0.35)
        repeats = rng.random((weeks, len(selected))) < repeat_probability[:, None]
        repeats &= np.arange(weeks)[:, None] >= first[None, :]
        events[edp][:, selected] |= repeats

    return Campaign(
        campaign_id=campaign_id,
        scenario=scenario,
        events=events,
        final_reach_fraction=reach,
        objectives=objectives,
        audience_strategies=audience_strategies,
    )
