"""Build the product-facing provider-model comparison notebook."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = NOTEBOOK_DIR / "meta_campaign_scenarios.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        r"""
# Provider-calibrated reach across realistic campaign scenarios

This notebook evaluates a simplified two-model architecture:

1. A **demographic-agnostic total-reach model** uses aggregate Reference-ID patterns, per-EDP reach, and optional campaign context. A model provider trains it against panel truth and freezes it for the model line.
2. The existing **VID demographic model** runs separately. The model provider supplies a packaged method that adjusts its age, gender, and geography results to add to the calibrated total.

The notebook compares this architecture with the existing VID baseline and with two measurement-layer calibration alternatives defined below. It reports total union-reach error and demographic-allocation error separately.

Synthetic truth is used only to evaluate the methods. The simulated reporting system never links a VID to a Reference ID.
"""
    ),
    markdown(
        r"""
## What changes in this architecture

One possible architecture asks the measurement system to translate Reference-ID overlaps into corrected pairwise and higher-order intersections. Here, that work is learned upstream by the model provider using its panel.

At runtime the measurement system performs three steps:

1. Run the provider's demographic-agnostic model to obtain calibrated total union reach.
2. Run the VID demographic model to obtain the initial demographic distribution.
3. Apply the provider-packaged demographic adjustment.

The provider may choose simple proportional scaling or a panel-learned adjustment. The measurement system applies the frozen choice and checks that the output is nonnegative, adds to the calibrated total, and remains within population limits.
"""
    ),
    markdown(
        r"""
## Synthetic campaign scenarios

The tests use plausible mechanisms rather than claims about Meta production traffic.

| Scenario | Objective or audience mechanism | Why it matters |
|---|---|---|
| Broad awareness | Broad awareness delivery | Negative control where the existing VID model should already work well |
| Traffic optimization | Click or landing-page optimization | Moderate audience selection |
| Engagement retargeting | Prior viewers or engagers | Warm audiences repeatedly selected across EDPs |
| Lead generation | Form, call, message, or sign-up optimization | Higher intent and potentially higher contactability |
| Sales prospecting | Purchase optimization | Shared conversion propensity across EDPs |
| Website retargeting | Site visitors or cart viewers | Small, highly overlapping audiences |
| Customer-list retargeting | First-party customer list | High overlap and unusually strong email matchability |
| Catalog retargeting | Product-view or catalog audiences | Strong pair-specific affinity |
| Lookalike prospecting | Audience modeled from a seed | Seed affinity combined with broader delivery |
| Audience expansion | Automated expansion beyond a seed | Blend of broad and selected delivery |
| App retargeting | Installs or in-app activity | High overlap but potentially lower email matchability |
| Unrelated niches | Different interests or exclusions | Checks that the model does not manufacture overlap |
| Mixed funnel | Different objectives across EDPs | Tests a realistic multi-objective report |

Every scenario is tested over early, partial, noncontiguous, and full-flight windows with 2, 5, and 10 EDPs.
"""
    ),
    code(
        r"""
from pathlib import Path
import csv
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "provider_model_final"

RUN_BENCHMARK = False
if RUN_BENCHMARK:
    from reference_calibration.provider_benchmark import run_provider_benchmark
    run_provider_benchmark(OUTPUT_DIR, profile="full")

summary = json.loads((OUTPUT_DIR / "provider_summary.json").read_text())
with (OUTPUT_DIR / "provider_metrics.csv").open() as stream:
    metrics = list(csv.DictReader(stream))

methods = {item["name"]: item for item in summary["methods"]}
print("Project: repository root")
print(f"Outputs: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
print(f"Panel-training campaigns: {summary['panel_train_campaigns']}")
print(f"Held-out panel campaigns: {summary['panel_holdout_campaigns']}")
print(f"Independent evaluation campaigns: {summary['evaluation_campaigns']}")
print(f"Demographic cells: {summary['demographic_cells']} (age × gender × geography)")
"""
    ),
    markdown(
        r"""
## Methods compared

The first three rows represent today's baseline and two measurement-layer approaches. The provider rows move total-reach calibration into a panel-trained model. The two oracle rows are diagnostics and are not implementable production methods.
"""
    ),
    code(
        r"""
method_rows = [
    {
        "method": item["label"],
        "type": item["category"],
        "description": item["description"],
    }
    for item in summary["methods"]
]
display(Markdown(markdown_table(
    method_rows,
    [("method", "Method"), ("type", "Type"), ("description", "What it does")],
)))
"""
    ),
    markdown(
        r"""
## Total union-reach result

The headline provider model uses Reference-ID measurements, EDP reaches, campaign scale, campaign objective, and audience strategy. The “Reference-ID inputs only” version removes objective and audience strategy to test whether that context adds useful information.
"""
    ),
    code(
        r"""
TOTAL_METHODS = [
    "existing_vid",
    "direct_pair_proportional",
    "mixture_pair_proportional",
    "provider_reference_proportional",
    "provider_context_proportional",
]

total_rows = []
for name in TOTAL_METHODS:
    item = methods[name]
    total_rows.append({
        "method": item["label"],
        "mean": f"{item['total_error']['mean']:.2%}",
        "p90": f"{item['total_error']['p90']:.2%}",
        "two": f"{item['total_error_by_edp_count']['2']['mean']:.2%}",
        "five": f"{item['total_error_by_edp_count']['5']['mean']:.2%}",
        "ten": f"{item['total_error_by_edp_count']['10']['mean']:.2%}",
    })

display(Markdown(markdown_table(
    total_rows,
    [
        ("method", "Method"),
        ("mean", "Mean total error"),
        ("p90", "p90"),
        ("two", "2 EDPs"),
        ("five", "5 EDPs"),
        ("ten", "10 EDPs"),
    ],
)))
display(Image(filename=str(OUTPUT_DIR / "provider_total_error_by_scenario.png")))
"""
    ),
    markdown(
        r"""
## Does campaign context help?

Yes in this synthetic design. The context-aware and context-free provider models see the same aggregate Reference-ID and reach measurements. The only difference is that the context-aware model also receives objective and audience-strategy summaries.

This should not be interpreted as proof that objective metadata will improve a real model by the same amount. It demonstrates how the hypothesis can be tested using whole-campaign panel holdouts.
"""
    ),
    code(
        r"""
context_rows = []
for name in ("provider_reference_proportional", "provider_context_proportional"):
    evaluation = methods[name]
    holdout = next(item for item in summary["holdout_methods"] if item["name"] == name)
    context_rows.append({
        "model": evaluation["label"],
        "evaluation": f"{evaluation['total_error']['mean']:.2%}",
        "holdout": f"{holdout['total_error']['mean']:.2%}",
        "parameters": summary["provider_models"][
            "context_parameter_count" if name == "provider_context_proportional"
            else "reference_only_parameter_count"
        ],
    })
display(Markdown(markdown_table(
    context_rows,
    [
        ("model", "Provider total model"),
        ("evaluation", "Independent evaluation error"),
        ("holdout", "Panel holdout error"),
        ("parameters", "Stored parameters"),
    ],
)))
"""
    ),
    markdown(
        r"""
## Demographic allocation options

Total accuracy and demographic accuracy are different problems. The provider total is held fixed below while the demographic adjustment changes:

- **Proportional scaling** preserves the existing VID demographic shares.
- **Fixed panel adjustment** applies one learned correction per demographic cell.
- **Contextual panel adjustment** changes the correction using objective, audience strategy, EDP mix, scale, and the original VID shares.

“Demographic distribution error” is the share of the reached audience that would need to move between the 18 age × gender × geography cells to match synthetic truth. The oracle rows use the true total reach and therefore isolate allocation error from total-reach error.
"""
    ),
    code(
        r"""
DEMO_METHODS = [
    "provider_context_proportional",
    "provider_context_fixed_demo",
    "provider_context_learned_demo",
    "oracle_total_proportional",
    "oracle_total_learned_demo",
]
demo_rows = []
for name in DEMO_METHODS:
    item = methods[name]
    demo_rows.append({
        "method": item["label"],
        "total": f"{item['total_error']['mean']:.2%}",
        "reach": f"{item['demographic_reach_error']['mean']:.2%}",
        "distribution": f"{item['demographic_distribution_error']['mean']:.2%}",
    })
display(Markdown(markdown_table(
    demo_rows,
    [
        ("method", "Method"),
        ("total", "Total error"),
        ("reach", "Combined demographic reach error"),
        ("distribution", "Demographic distribution error"),
    ],
)))
display(Image(filename=str(OUTPUT_DIR / "provider_demographic_error_by_scenario.png")))
"""
    ),
    markdown(
        r"""
## Selected campaign examples

The table below includes the broad-awareness control, high-overlap retargeting cases, the low-email app case, and the mixed-objective portfolio. It compares the existing VID result with the full provider architecture: context-aware total reach plus context-aware demographic adjustment.
"""
    ),
    code(
        r"""
spotlights = [
    "broad_awareness_control",
    "website_retargeting",
    "crm_customer_list",
    "app_activity_retargeting",
    "mixed_funnel_portfolio",
]
spotlight_rows = []
for scenario in spotlights:
    baseline = summary["scenario_summary"][scenario]["existing_vid"]
    provider = summary["scenario_summary"][scenario]["provider_context_learned_demo"]
    spotlight_rows.append({
        "scenario": next(
            row["scenario_label"]
            for row in metrics
            if row["scenario"] == scenario
        ),
        "baseline_total": f"{baseline['total_error']['mean']:.1%}",
        "provider_total": f"{provider['total_error']['mean']:.1%}",
        "baseline_demo": f"{baseline['demographic_distribution_error']['mean']:.1%}",
        "provider_demo": f"{provider['demographic_distribution_error']['mean']:.1%}",
    })
display(Markdown(markdown_table(
    spotlight_rows,
    [
        ("scenario", "Scenario"),
        ("baseline_total", "Existing VID total error"),
        ("provider_total", "Provider total error"),
        ("baseline_demo", "Existing VID demographic error"),
        ("provider_demo", "Provider demographic error"),
    ],
)))
"""
    ),
    markdown(
        r"""
## Cross-report consistency

Each model is evaluated independently for every requested window and EDP set. The checks below compare nested requests, such as weeks 1–3 versus weeks 1–12 and two EDPs versus five or ten EDPs.

The provider model substantially improves accuracy, but independent predictions can still produce occasional monotonicity violations. The existing stored-result reconciliation layer therefore remains applicable. Moving calibration into the provider model simplifies report-time calibration; it does not by itself guarantee consistency among every report requested at different times.
"""
    ),
    code(
        r"""
consistency_rows = []
for name in TOTAL_METHODS:
    result = summary["consistency"]["evaluation"][name]
    consistency_rows.append({
        "method": methods[name]["label"],
        "checks": result["checks"],
        "violations": result["violations"],
        "maximum": f"{result['maximum'] / 1_000_000:.2f}M",
    })
display(Markdown(markdown_table(
    consistency_rows,
    [
        ("method", "Method"),
        ("checks", "Nested-report checks"),
        ("violations", "Raw violations"),
        ("maximum", "Largest violation"),
    ],
)))
"""
    ),
    markdown(
        r"""
## Interpretation

The synthetic results support the architecture as a serious candidate:

- Moving calibration into a panel-trained provider model performs better here than fitting overlap corrections only from broad-reach campaigns.
- Campaign objective and audience strategy materially improve the synthetic provider model, especially for retargeting and app cases.
- Correct total reach does not automatically correct demographics. Proportional scaling preserves a meaningful demographic error even with oracle total reach.
- A panel-learned demographic adjustment can reduce that error, but it should remain the model provider's choice and must beat proportional scaling on whole-campaign holdouts.
- The broad-awareness and mixed-funnel controls matter: a more complex model is not automatically better for every report.
- The provider model still needs the existing cross-report reconciliation layer when published results must remain consistent over time.

These numbers are evidence that the interfaces and test design work, not production accuracy claims. The decisive next step is to repeat the comparison using approved aggregate observations and model-provider panel holdouts.
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
