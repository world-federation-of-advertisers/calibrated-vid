# Reference-ID calibration synthetic validation

This package is a reproducible synthetic implementation of alternative ways to
use Reference IDs to improve cross-EDP reach. It implements:

- a single common Reference-ID join key hashed into a 5-billion-value pool;
- EDP-specific proprietary fallback identifiers that match across EDPs only
  through random pool collisions;
- heterogeneous email availability and approximately 60% conditional email
  agreement, with stable pair-specific affinity;
- complete pairwise-through-N intersection measurement for as many as 10 EDPs;
- a distinct 5,000-person panel, separate from the synthetic evaluation truth;
- representative, low-effective-size, observably biased, and hidden-bias panel designs;
- a panel-trained, email-first demographic-agnostic VID response that receives
  email and proprietary identifiers separately, directly anchors shared-email
  VIDs, and uses optional campaign context only for non-email behavior;
- a provider-trained aggregate combiner that can use both VID outputs without
  linking their person-level VIDs;
- an imperfect VID demographic model over 18 age × gender × geography cells;
- proportional, fixed, and contextual demographic adjustment methods;
- the pair-aware fixed-plus-log calibration family;
- the two-group latent-matchability calibration family;
- nonnegative exclusive-cell decoding for internally valid union reach;
- stored-result reconciliation for reports over different EDP and week sets;
- production of a bounded `REVIEW_REQUIRED` result when prior results cannot be
  reconciled exactly.

The calibration Reference ID is never an input to either VID model. The
simulation never links a VID to a Reference ID. Synthetic truth is retained
only by the test harness so accuracy can be measured.

`VALIDATION.md` documents the original measurement-layer calibration and
stored-result reconciliation experiment. The two notebooks evaluate four
input combinations produced by two independent choices: whether the provider
supplies only the demographic VID output or both VID outputs, and whether its
finalization function may use aggregate Reference-ID overlap.

For a product-grounded walkthrough, open
`notebooks/meta_campaign_scenarios.ipynb`. It compares demographic VID only,
both VID models, demographic VID plus Reference-ID correction, and both VID
models plus Reference-ID correction
across plausible Meta campaign types:
traffic, engagement retargeting, leads, sales, website and customer-list
retargeting, catalog campaigns, lookalikes, Advantage+ audience expansion, app
activity, unrelated niches, and a mixed-funnel portfolio. The notebook is
already executed. It reports total union-reach and demographic-allocation error
separately and includes early, partial, noncontiguous, and full-flight reports
with 2, 5, and 10 EDPs.

Launch the notebook locally with:

```bash
.venv/bin/python -m jupyterlab notebooks/meta_campaign_scenarios.ipynb
```

The technical model comparison is in
`notebooks/calibration_method_benchmark.ipynb`. It explains the distinct
5,000-person panel; the provider-owned fitting and selection flow; sampling and
selection error; the impression-level information boundary; separate
Reference-ID selection for the demographic-only and two-VID functions;
demographic allocation; and raw
cross-report consistency. Re-run the underlying experiment with:

```bash
PYTHONPATH=src .venv/bin/python -m reference_calibration.panel_validation \
  --profile full \
  --output-dir outputs/panel_5000_final
```

## Quick start

From the repository root:

```bash
./scripts/bootstrap.sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/reference-calibration-sim --profile quick --output-dir outputs/quick
```

The bootstrap script creates an isolated virtual environment and installs the
simulation and notebook dependencies.

## Full validation

```bash
.venv/bin/reference-calibration-sim --profile full --output-dir outputs/full
.venv/bin/python -m reference_calibration.panel_validation \
  --profile full --output-dir outputs/panel_5000_final
```

The first command reproduces the original measurement-layer experiment. The
second reproduces the four-input-combination provider package and
demographic-allocation comparison.
The full profiles take longer because they use more campaigns, users, report
shapes, and stress cases. The original experiment writes:

- `summary.json`: machine-readable configuration and headline metrics;
- `model_artifacts.json`: fitted parameters and version metadata for both
  calibration options;
- `metrics.csv`: detailed accuracy and consistency measurements;
- `validation_report.md`: readable findings and acceptance checks;
- `error_by_scenario.png`: baseline and calibrated error comparison.
- `linkage_shift_sweep.png`: break-even behavior as audience matchability shifts.

The panel benchmark writes `panel_validation_summary.json`, detailed metrics,
panel-draw and activation-decision files, an example set of frozen
provider-to-measurement instructions in `provider_packages.json`, and accuracy charts to
`outputs/panel_5000_final/`.

The label-time research spike is in
`notebooks/daily_match_labeling_research.ipynb`. It asks whether same-day or
campaign-to-date Reference-ID matching can be encoded into immutable VID
labels. It tests adaptive hash pools, stored first-seen mappings,
collision-resolved 1:1 assignment, fixed-marginal and pair-targeted overlap
allocators, pair and higher-order rank lattices,
buffered labeling, cross-campaign map scope, and oracle online union/Venn
allocators. Reproduce it with:

```bash
PYTHONPATH=src .venv/bin/python -m reference_calibration.daily_labeling \
  --profile full --output-dir outputs/daily_labeling_final
PYTHONPATH=src .venv/bin/python -m reference_calibration.venn_information_proof \
  --profile full --output-dir outputs/full_venn_proof_final
.venv/bin/python notebooks/build_daily_match_labeling_research.py
```

## What is validated

The original calibration experiment fits on large, broadly targeted campaigns.
The provider experiment trains on repeated 5,000-person panel draws containing
all tested campaign objectives and audience strategies. Separate campaigns are
used to train the email-first demographic-agnostic model, learn the aggregate
two-VID combiner, fit Reference-ID response candidates, select corrections and
the complete provider recommendation, and evaluate the frozen choice. The
agnostic-only output is retained as a diagnostic, while the deployment
comparison asks whether making both VID outputs available improves the final
provider function. It tests both sampling noise and panel-selection bias. In
both experiments, entire campaigns are assigned to fitting, holdout, or
independent evaluation splits.
Stress and evaluation campaigns include:

- a small non-reach campaign compared with larger reach campaigns;
- two small correlated campaigns;
- two small disjoint campaigns;
- several small correlated campaigns;
- mixed reach and non-reach objectives;
- unusually high-matchability remarketing-like audiences;
- unusually low-matchability audiences.

Report requests include weeks 1-3 followed by weeks 1-12, weeks 5-12 for two
EDPs, weeks 7-13 for five EDPs, full-flight reports, noncontiguous week sets,
and reports containing 2, 5, and 10 EDPs. Request-order permutations quantify
the remaining order sensitivity of stored-result reconciliation.

The provider experiment also measures demographic accuracy for 18 mutually
exclusive age × gender × geography cells. It compares proportional scaling
with fixed and context-dependent panel adjustments.

## Important interpretation

This is a synthetic validation harness, not evidence that either model is
correct for production data. Its purpose is to verify the mathematics,
interfaces, scaling behavior, failure handling, and test methodology. Model
selection should ideally be repeated on approved aggregate observations from
real calibration campaigns.
