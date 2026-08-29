"""Build the technical 5,000-person provider-package validation notebook."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = NOTEBOOK_DIR / "calibration_method_benchmark.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        r"""
# Technical validation of the provider-finalized reach package

The proposed design has two independent choices:

1. **VID outputs:** supply only the demographic VID output or supply both demographic and demographic-agnostic VID outputs.
2. **Reference-ID input:** do not use aggregate Reference-ID overlap or allow the provider's finalization function to use it.

Together they produce four input combinations:

| VID outputs supplied | Without Reference-ID overlap | With Reference-ID overlap |
|---|---|---|
| Demographic VID only | Provider instructions return the existing VID result | Provider instructions combine demographic VID aggregates with Reference-ID overlaps |
| Demographic and demographic-agnostic VID | Provider instructions may use either or both VID aggregate outputs | Provider instructions may use either or both VID aggregate outputs and Reference-ID overlaps |

The provider can train, select, and validate the full package because it has panel-person truth. It then publishes one explicit total-reach finalization function with the model line. The measurement-system operator runs the required VID model or models, computes approved aggregate Reference-ID intersections inside the TEE when needed, applies the frozen function, and enforces output bounds. Operator-side fitting is possible if the operator later obtains adequate truth, but it is not required by this design.

An **Event Data Provider (EDP)** is a publisher or other data source contributing campaign events. The VID labeler receives email and EDP-proprietary identifiers separately. The optional demographic-agnostic labeler can use a shared email as a direct cross-EDP VID anchor and may use objective, audience strategy, co-viewing, or other permitted context for proprietary-ID and ambiguous cases. Separately, the calibration workload derives a **Reference ID** from normalized email when available and otherwise from that EDP's proprietary identifier, hashed into a shared 5-billion-value space. Reference ID is not a VID-labeler input. “RID” is used only as a compact label in tables and charts.
"""
    ),
    code(
        r"""
from pathlib import Path
import csv
import json
import sys

from IPython.display import Image, Markdown, display


def find_project_root():
    for candidate in (Path.cwd(), Path.cwd().parent):
        if (candidate / "src" / "reference_calibration").exists():
            return candidate.resolve()
    raise RuntimeError("Run this notebook from the repository root or notebooks/.")


def markdown_table(rows, columns):
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(row[key]).replace("|", "/") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "panel_5000_final"

RUN_BENCHMARK = False
if RUN_BENCHMARK:
    from reference_calibration.panel_validation import run_panel_validation
    run_panel_validation(OUTPUT_DIR, profile="quick")

summary = json.loads((OUTPUT_DIR / "panel_validation_summary.json").read_text())
with (OUTPUT_DIR / "panel_validation_metrics.csv").open() as stream:
    metrics = list(csv.DictReader(stream))
with (OUTPUT_DIR / "panel_draws.csv").open() as stream:
    panel_draws = list(csv.DictReader(stream))
with (OUTPUT_DIR / "activation_decisions.csv").open() as stream:
    decisions = list(csv.DictReader(stream))
provider_packages = json.loads((OUTPUT_DIR / "provider_packages.json").read_text())
"""
    ),
    markdown(
        r"""
## 1. Information boundary and frozen artifacts

VID models are impression-level mappings. They receive email and EDP-proprietary identifiers as separate inputs. Depending on the approved design, the demographic-agnostic model may also use EDP, campaign objective, audience strategy, co-viewing, and other impression-available context when assigning VIDs for non-email cases. It does not receive the calibration Reference ID, report-level campaign size, or aggregate cross-EDP intersections.

The Reference-ID calibrator is not a third VID model. It is an optional part of a frozen finalization function that consumes aggregate report measurements. The provider can make that function use either VID output, both VID outputs, Reference-ID overlap, or a validated combination of them. No person-level VID link between the two models—and no VID-to-Reference-ID link—is required.

A deployable provider package therefore contains whichever of the following are selected for a model line:

- the current demographic-ready VID model;
- the optional demographic-agnostic VID model;
- one provider-supplied total-reach finalization function, including any optional Reference-ID response and decoder instructions; and
- the demographic-adjustment instructions that map the selected total to the demographic-ready VID distribution.
"""
    ),
    code(
        r"""
package = provider_packages[0]


def finalization_summary(item):
    inputs = ", ".join(item["vid_inputs"])
    rid = "with Reference-ID overlap" if item["uses_reference_id"] else "without Reference-ID overlap"
    return f"{inputs}; {rid}"


display(Markdown(markdown_table(
    [
        {
            "item": "Agnostic VID labeler inputs",
            "value": ", ".join(package["vid_models"]["demographic_agnostic"]["identity_inputs"]),
        },
        {
            "item": "Reference ID used by labeler?",
            "value": str(package["vid_models"]["demographic_agnostic"]["uses_reference_id_calibration_input"]),
        },
        {
            "item": "Selected total-reach function",
            "value": finalization_summary(package["selected_total_reach_function"]),
        },
        {
            "item": "Person-level cross-model link used?",
            "value": str(package["selected_total_reach_function"]["uses_person_level_crosswalk"]),
        },
    ],
    [("item", "Frozen package item"), ("value", "Example value")],
)))
"""
    ),
    markdown(
        r"""
## 2. Synthetic training, selection, and evaluation flow

The experiment creates a large synthetic population with exact person-level truth. That truth is retained only by the test harness.

For each 5,000-person panel draw:

1. Train the email-first demographic-agnostic response on one group of campaigns, with email and proprietary identifiers kept separate and optional context used only for non-email behavior.
2. On a second group, learn an aggregate two-VID combiner and fit candidate Reference-ID response functions from weighted panel-person truth and aggregate panel Reference-ID intersections.
3. On a third group of whole campaigns, separately select a Reference-ID correction for the demographic-only and two-VID functions, or select identity for either path.
4. On the same held-out selection group, choose among the four input combinations, retaining existing VID unless an alternative clears the guardrails.
5. Freeze one total-reach finalization function and score it on a fourth, independent campaign group using full-population truth.

All splits are by campaign. Weekly snapshots from one campaign never appear in both fitting and validation data.
"""
    ),
    markdown(
        r"""
## 3. What the two-VID path represents

Reproducing a provider's proprietary impression-to-VID learner is outside this harness. Instead, the benchmark reconstructs the aggregate pair-duplication response that a frozen email-first agnostic model might produce.

For each EDP pair, observed shared-email overlap is treated as a direct VID anchor. A fitted response then estimates only the remaining overlap associated with proprietary identifiers, co-viewing, and other non-email cases, using EDP identities and optional objective or audience-strategy context. At report time, those pair relationships and the per-EDP reaches are combined into one valid multi-EDP audience. Ten EDPs require 45 pair responses plus pooled context effects, rather than 1,023 unrelated subset curves.

The provider must then turn the two VID outputs into one result. The synthetic implementation learns one number, an agnostic-model weight between zero and one. For every EDP pair, it blends the demographic-model and agnostic-model intersection estimates using that same weight, then reconstructs one valid multi-EDP audience. A weight of zero means the provider should use only the demographic model; a weight of one means it should use only the agnostic model; an interior value uses both.

This deliberately simple combiner is easy to inspect and requires only aggregate intersections from the two models. It does not link their person-level VIDs, link a VID to a calibration Reference ID, or claim that a production provider must use this exact one-parameter form. The agnostic-only result remains in the benchmark as a diagnostic, but the deployment comparison asks whether supplying both outputs improves the provider's final answer.
"""
    ),
    markdown(
        r"""
## 4. Campaign test matrix

The benchmark uses stylized versions of campaign objectives and audience tools available in products such as Meta Ads Manager. They are designed to cover distinct measurement mechanisms, not to estimate Meta's production audiences or match rates.

- **Campaign size** controls how much direct panel evidence is available.
- **Audience overlap across EDPs** controls how many of the same people are truly reached by multiple publishers.
- **Shared-email visibility** controls how much true duplication can be anchored directly by email in the agnostic VID model and, separately, how much appears in calibration Reference-ID intersections.

These properties are varied separately. This prevents the experiment from assuming that an objective name—such as “Sales”—automatically determines either true overlap or email matchability. Across the ten EDPs, base email availability ranges from 10% to 95% and the email-agreement parameter ranges from 52% to 72%, centered near 60%; the scenarios then select audiences with higher or lower matchability than that base population.
"""
    ),
    code(
        r"""
scenario_rows = []
for item in summary["scenario_descriptions"].values():
    scenario_rows.append({
        "objective": item["objective"],
        "setup": item["audience"],
        "size": item["volume"],
        "overlap": item["cross_edp_similarity"],
        "email": item["reference_matchability"],
        "purpose": item["intuition"],
    })
display(Markdown(markdown_table(
    scenario_rows,
    [
        ("objective", "Objective"),
        ("setup", "Real-world campaign setup"),
        ("size", "Relative size"),
        ("overlap", "Expected audience overlap across EDPs"),
        ("email", "Expected shared-email visibility"),
        ("purpose", "What the scenario tests"),
    ],
)))
"""
    ),
    markdown(
        r"""
## 5. Provider-fitted Reference-ID candidates

The provider can observe, for panel campaigns, both weighted panel-person truth and aggregate Reference-ID intersections. That is enough to learn how much true overlap is visible through the common Reference ID without linking an existing VID to a Reference ID.

The benchmark fits three candidate families:

- **Fixed capture:** stable pair-specific visibility rates with no campaign-size term. This is easy to estimate and may transfer well when matching behavior is stable.
- **Fixed plus log scale:** pair-specific rates plus a pre-fitted logarithmic campaign-size effect. This can capture a smooth size relationship but may overfit or transfer poorly.
- **Two-group mixture:** a compact representation of people who are consistently easier or harder to match across EDPs. This can capture person-level correlation in email provision.

Each candidate uses observed pairwise, three-way, four-way, and higher-order panel intersections during fitting. At runtime, the current implementation estimates pairwise targets and infers higher orders through one internally consistent audience reconstruction (implemented here with maximum entropy). The result is one set of non-overlapping Venn regions, none of which can contain a negative audience.

The selected candidate is not merely a research conclusion. The provider publishes it as a frozen instruction bundle for the downstream measurement service: which base VID result to use, the fitted coefficients, how to subtract the 5B collision floor, how to decode a valid audience, the supported range, diagnostics, and the identity/fallback rule. The measurement service supplies the new report's aggregate VID and Reference-ID measurements to that bundle.
"""
    ),
    markdown(
        r"""
## 6. Panel designs

The raw panel size is fixed at 5,000. Effective size can be smaller when weights are unequal. Observable recruitment bias is corrected using known weights; hidden selection on intent and matchability is deliberately left uncorrected. The hidden-bias case is intentionally severe so the failure mode is visible, not a production prevalence estimate. This executed notebook uses four panel draws per condition, so its selection rates illustrate behavior rather than estimate stable production probabilities.
"""
    ),
    code(
        r"""
panel_rows = []
for design, item in summary["panel_designs"].items():
    panel_rows.append({
        "panel": item["label"],
        "raw": item["raw_size"],
        "mean_neff": f"{item['effective_size']['mean']:.0f}",
        "min_neff": f"{min(float(row['effective_size']) for row in panel_draws if row['panel_design'] == design):.0f}",
        "description": item["description"],
    })
display(Markdown(markdown_table(
    panel_rows,
    [("panel", "Panel"), ("raw", "Raw N"), ("mean_neff", "Mean effective N"), ("min_neff", "Minimum effective N"), ("description", "Purpose")],
)))
"""
    ),
    code(
        r"""
panel_truth_rows = []
band_labels = {
    "small_under_10_percent": "Small (<10%)",
    "medium_10_to_30_percent": "Medium (10%–30%)",
    "large_over_30_percent": "Large (>30%)",
}
for design, bands in summary["panel_truth_summary"].items():
    for band, result in bands.items():
        panel_truth_rows.append({
            "panel": summary["panel_designs"][design]["label"],
            "volume": band_labels[band],
            "mean": f"{result['mean']:.2%}",
            "p90": f"{result['p90']:.2%}",
            "worst": f"{result['max']:.2%}",
        })
display(Markdown("### Panel-estimated truth versus full synthetic truth\n\n" + markdown_table(
    panel_truth_rows,
    [("panel", "Panel"), ("volume", "Volume"), ("mean", "Mean error"), ("p90", "p90"), ("worst", "Worst")],
)))
"""
    ),
    markdown(
        r"""
## 7. Selection rule

Selection happens in two stages so the design answers two distinct questions.

First, for the demographic-only and two-VID functions separately, a Reference-ID family must:

- improve mean absolute relative error by at least 0.5 percentage points;
- avoid worsening p90 error by more than 0.5 percentage points; and
- show, after accounting for campaign-to-campaign variability, at least 90% confidence that its average error is lower.

If no family passes, that path uses its VID-only function—no Reference-ID correction.

Second, the provider applies the same guardrails to the four input combinations, always keeping existing VID as the fallback. This means the provider can recommend the demographic-only or two-VID function, with or without Reference-ID overlap.
"""
    ),
    code(
        r"""
configuration_labels = {
    "existing_vid": "Existing VID",
    "two_vid": "Both VID models",
    "existing_plus_selected_rid": "Existing VID + selected RID",
    "two_vid_plus_selected_rid": "Both VID models + selected RID",
}
activation_rows = []
for design, item in summary["activation_summary"].items():
    activation_rows.append({
        "panel": summary["panel_designs"][design]["label"],
        "existing_active": f"{item['existing_correction_active_rate']:.0%}",
        "existing_harm": f"{item['existing_correction_harm_rate']:.0%}",
        "two_vid_active": f"{item['two_vid_correction_active_rate']:.0%}",
        "two_vid_harm": f"{item['two_vid_correction_harm_rate']:.0%}",
        "agnostic_weight": f"{item['two_vid_agnostic_weight']['mean']:.0%}",
        "recommendations": ", ".join(
            f"{configuration_labels[name]}: {count}"
            for name, count in item["recommended_configuration_counts"].items()
            if count
        ),
    })
display(Markdown(markdown_table(
    activation_rows,
    [
        ("panel", "Panel"),
        ("existing_active", "RID active on existing"),
        ("existing_harm", "Existing-base activation harmed truth"),
        ("two_vid_active", "RID active on two-VID function"),
        ("two_vid_harm", "Two-VID activation harmed truth"),
        ("agnostic_weight", "Mean agnostic weight"),
        ("recommendations", "Complete configuration selected"),
    ],
)))
"""
    ),
    markdown(
        r"""
## 8. Accuracy of the four input combinations

The table aggregates all independent evaluation campaigns and report shapes. The “selected RID” rows use the family chosen for that VID path, or the VID-only function if no family passed. The provider-recommended row uses the finalization function chosen on panel holdouts. The p90 and p99 columns are the error levels that 90% and 99% of tested reports are at or below. Relative error can exceed 100% when the true audience is small; a 140% error means the estimate missed by 1.4 times the true reach.

The existing-VID baseline is intentionally generated from a population-rate overlap assumption. It is almost exact for the broad-awareness control and can be severely wrong for narrow, correlated campaigns. These errors are useful for comparing methods inside the synthetic world, but they are not estimates of production VID accuracy.

“Provider recommended” is not a fifth alternative. It is the finalization function selected from the four input combinations in each panel draw.
"""
    ),
    code(
        r"""
METHODS = [
    "existing_vid",
    "two_vid",
    "existing_plus_selected_rid",
    "two_vid_plus_selected_rid",
    "provider_recommended",
]
method_labels = {
    **configuration_labels,
    "provider_recommended": "Provider recommended",
}
rows = []
for design, values in summary["method_summary"].items():
    for method in METHODS:
        result = values[method]
        rows.append({
            "panel": summary["panel_designs"][design]["label"],
            "method": method_labels[method],
            "mean": f"{result['mean']:.2%}",
            "p90": f"{result['p90']:.2%}",
            "p99": f"{result['p99']:.2%}",
            "max": f"{result['max']:.2%}",
        })
display(Markdown(markdown_table(
    rows,
    [("panel", "Panel"), ("method", "Configuration"), ("mean", "Mean"), ("p90", "p90"), ("p99", "p99"), ("max", "Worst")],
)))
display(Image(filename=str(OUTPUT_DIR / "error_by_panel_design.png")))
"""
    ),
    markdown(
        r"""
### Agnostic-only diagnostic

The two-VID deployment path may use either or both aggregate outputs. This harness tests one deliberately simple finalization function: a single bounded blend of the two models' pair-intersection estimates. The agnostic-only result below is retained as a diagnostic. When it beats the learned blend, the extra model contains useful information but the one-parameter combiner has not extracted all of it.
"""
    ),
    code(
        r"""
diagnostic_rows = []
for design, values in summary["method_summary"].items():
    diagnostic_rows.append({
        "panel": summary["panel_designs"][design]["label"],
        "agnostic": f"{values['agnostic_vid_diagnostic']['mean']:.2%}",
        "two_vid": f"{values['two_vid']['mean']:.2%}",
        "weight": f"{summary['activation_summary'][design]['two_vid_agnostic_weight']['mean']:.1%}",
    })
display(Markdown(markdown_table(
    diagnostic_rows,
    [
        ("panel", "Panel"),
        ("agnostic", "Agnostic-only error"),
        ("two_vid", "Two-VID function error"),
        ("weight", "Agnostic weight"),
    ],
)))
"""
    ),
    code(
        r"""
hidden = summary["method_summary"]["hidden_matchability_bias"]
hidden_choice = summary["activation_summary"]["hidden_matchability_bias"]["recommended_configuration_counts"]
draw_count = sum(hidden_choice.values())
choice_text = ", ".join(
    f"**{method_labels[name]}** in {count} of {draw_count} draws"
    for name, count in hidden_choice.items()
    if count
)
display(Markdown(
    "**Hidden-bias stress result.** The panel selected "
    f"{choice_text}. The recommended configuration's full-population mean error was "
    f"**{hidden['provider_recommended']['mean']:.2%}**; for comparison, the two-VID function had "
    f"**{hidden['two_vid']['mean']:.2%}** error and the agnostic-only diagnostic had "
    f"**{hidden['agnostic_vid_diagnostic']['mean']:.2%}** error. The selection process behaved as designed; "
    "the panel was missing an important trait that affected the full population."
))
"""
    ),
    markdown(
        r"""
### Which correction family was selected for each VID path?

These counts are intentionally separate. A correction that helps the demographic VID model can be redundant or harmful after the two-VID function has already removed part of the same error.
"""
    ),
    code(
        r"""
family_rows = []
family_labels = {
    "existing_vid": "No correction",
    "two_vid": "No correction",
    "existing_plus_fixed": "Fixed capture",
    "two_vid_plus_fixed": "Fixed capture",
    "existing_plus_fixed_log": "Fixed + log size",
    "two_vid_plus_fixed_log": "Fixed + log size",
    "existing_plus_mixture": "Two-group mixture",
    "two_vid_plus_mixture": "Two-group mixture",
}
for design, item in summary["activation_summary"].items():
    family_rows.append({
        "panel": summary["panel_designs"][design]["label"],
        "existing": ", ".join(
            f"{family_labels[name]}: {count}" for name, count in item["existing_correction_selection_counts"].items() if count
        ),
        "two_vid": ", ".join(
            f"{family_labels[name]}: {count}" for name, count in item["two_vid_correction_selection_counts"].items() if count
        ),
    })
display(Markdown(markdown_table(
    family_rows,
    [("panel", "Panel"), ("existing", "Existing-VID selection"), ("two_vid", "Two-VID selection")],
)))
"""
    ),
    markdown(
        r"""
## 9. Interpreting the 5,000-person limit

For a simple random panel of effective size $N$, a rough relative 95% sampling interval for a reach proportion $p$ is:

$$
1.96\sqrt{p(1-p)/N}/p.
$$

This approximation ignores design effects and hidden selection bias, so it is optimistic for weighted or non-representative panels. Higher-order intersections are much sparser than single-EDP reach. If a true intersection is 0.02% of the population, a 5,000-person panel expects one matching person and has about a 37% chance of observing none.

The practical implication is that the panel should train and validate pooled behavior across many campaigns. It should not be treated as precise ground truth for every individual low-volume campaign.
"""
    ),
    code(
        r"""
sampling_rows = []
for proportion in (0.20, 0.10, 0.05, 0.01, 0.005, 0.002, 0.0002):
    expected = 5_000 * proportion
    relative = 1.96 * ((proportion * (1 - proportion) / 5_000) ** 0.5) / proportion
    zero = (1 - proportion) ** 5_000
    sampling_rows.append({
        "proportion": f"{proportion:.2%}",
        "expected": f"{expected:.1f}",
        "relative": f"±{relative:.0%}",
        "zero": f"{zero:.0%}",
    })
display(Markdown(markdown_table(
    sampling_rows,
    [("proportion", "Reach/intersection"), ("expected", "Expected panel people"), ("relative", "Approx. relative 95% interval"), ("zero", "Chance of zero observed")],
)))
"""
    ),
    markdown(
        r"""
## 10. Demographic adjustment is downstream of total reach

The measurement system always runs the demographic-ready model. When a different total is selected, the provider's demographic instruction adjusts that starting distribution so it adds to the selected total while respecting population limits.

The benchmark compares proportional scaling with a panel-learned contextual adjustment. Demographic-distribution error measures how far the reported shares differ from the true shares. Per-cell reach error also penalizes getting the total size of each demographic bucket wrong. Lower is better for both. Reference-ID overlaps supply no age, gender, or geography labels, so they cannot directly determine which demographic bucket changes.
"""
    ),
    code(
        r"""
demo_methods = (
    "existing_vid",
    "two_vid_proportional_demo",
    "two_vid_panel_demo",
    "existing_rid_panel_demo",
    "two_vid_rid_panel_demo",
    "recommended_panel_demo",
)
demo_labels = {
    "existing_vid": "Existing VID demographics",
    "two_vid_proportional_demo": "Two-VID total + proportional scaling",
    "two_vid_panel_demo": "Two-VID total + panel adjustment",
    "existing_rid_panel_demo": "Existing + RID + panel adjustment",
    "two_vid_rid_panel_demo": "Two-VID + RID + panel adjustment",
    "recommended_panel_demo": "Recommended total + panel adjustment",
}
demo_rows = []
for design in summary["panel_designs"]:
    for method in demo_methods:
        distribution_values = [
            float(row["value"])
            for row in metrics
            if row["panel_design"] == design
            and row["category"] == "demographic_distribution_error"
            and row["method"] == method
        ]
        reach_values = [
            float(row["value"])
            for row in metrics
            if row["panel_design"] == design
            and row["category"] == "demographic_reach_error"
            and row["method"] == method
        ]
        demo_rows.append({
            "panel": summary["panel_designs"][design]["label"],
            "method": demo_labels[method],
            "distribution": f"{sum(distribution_values) / len(distribution_values):.2%}",
            "reach": f"{sum(reach_values) / len(reach_values):.2%}",
        })
display(Markdown(markdown_table(
    demo_rows,
    [("panel", "Panel"), ("method", "Demographic output"), ("distribution", "Distribution error"), ("reach", "Per-cell reach error")],
)))
"""
    ),
    markdown(
        r"""
## 11. Cross-report consistency

Every configuration is tested on nested requests, including weeks 1–3 followed by weeks 1–12 and 2-, 5-, and 10-EDP subsets. The decoder guarantees a valid Venn diagram inside each report. It does not guarantee that reports calculated separately will agree wherever their weeks and EDPs overlap.

Freezing the model line and provider bundle makes a request reproducible without retrieving a prior result. Stored results are needed only when a new overlapping request must be checked against what has already been published. The raw violation rate below measures that remaining need for reconciliation.
"""
    ),
    code(
        r"""
consistency_rows = []
for design, item in summary["activation_summary"].items():
    consistency_rows.append({
        "panel": summary["panel_designs"][design]["label"],
        "violation_rate": f"{item['raw_consistency_violation_rate']:.2%}",
    })
display(Markdown(markdown_table(consistency_rows, [("panel", "Panel"), ("violation_rate", "Raw nested-report violation rate")])))
"""
    ),
    markdown(
        r"""
## 12. What this benchmark can and cannot establish

It establishes that all four input combinations can share one operator interface; that a provider can combine two VID aggregate outputs without linking their person-level identifiers; that optional correction can be learned from panel-only truth and aggregate Reference-ID observations; and that the resulting frozen package can be tested across 2-, 5-, and 10-EDP reports and varied week windows. It also quantifies sampling variability, panel-selection risk, correction-selection error, demographic allocation, and raw cross-report consistency in a controlled world.

It does not reproduce a provider's full production VID learner, prove that a real 5,000-person panel represents small retargeting or conversion audiences, or show that the synthetic winner will be the production winner. The hidden-bias experiment also shows a key limitation: a provider can make a statistically disciplined choice using its panel and still select the wrong configuration for the full population if the panel omits an important selection mechanism.

Production validation should therefore use repeated panel resampling, whole-campaign holdouts, results by campaign mechanism rather than one pooled score, and external truth where available. Shadow Reference-ID measurement remains useful for implementation checks and drift monitoring even when the provider supplies the initial bundle.
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbf.write(notebook, OUTPUT_PATH)
print(OUTPUT_PATH)
