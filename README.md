# Reference-ID calibration synthetic validation

This package is a reproducible synthetic implementation of alternative ways to
use Reference IDs to improve cross-EDP reach. It implements:

- a single common Reference-ID join key hashed into a 5-billion-value pool;
- EDP-specific proprietary fallback identifiers that match across EDPs only
  through random pool collisions;
- heterogeneous email availability and approximately 60% conditional email
  agreement, with stable pair-specific affinity;
- complete pairwise-through-N intersection measurement for as many as 10 EDPs;
- a panel-trained, demographic-agnostic total-reach model;
- optional campaign-objective and audience-strategy inputs;
- an imperfect VID demographic model over 18 age × gender × geography cells;
- proportional, fixed, and contextual demographic adjustment methods;
- the pair-aware fixed-plus-log calibration family;
- the two-group latent-matchability calibration family;
- nonnegative exclusive-cell decoding for internally valid union reach;
- stored-result reconciliation for reports over different EDP and week sets;
- production of a bounded `REVIEW_REQUIRED` result when prior results cannot be
  reconciled exactly.

The simulation never links a VID to a Reference ID. Synthetic truth is retained
only by the test harness so accuracy can be measured.

`VALIDATION.md` documents the original measurement-layer calibration and
stored-result reconciliation experiment. The two notebooks evaluate the newer
provider-owned total-reach and demographic-adjustment architecture.

For a product-grounded walkthrough, open
`notebooks/meta_campaign_scenarios.ipynb`. It compares the existing VID
baseline, measurement-layer calibration, and the provider-model architecture
across plausible Meta campaign types:
traffic, engagement retargeting, leads, sales, website and customer-list
retargeting, catalog campaigns, lookalikes, Advantage+ audience expansion, app
activity, unrelated niches, and a mixed-funnel portfolio. The notebook is
already executed. It reports total union-reach and demographic-allocation error
separately and includes early, partial, noncontiguous, and full-flight reports
with 2, 5, and 10 EDPs.

Launch the notebook locally with:

```bash
.venv/bin/jupyter lab notebooks/meta_campaign_scenarios.ipynb
```

The technical model comparison is in
`notebooks/calibration_method_benchmark.ipynb`. It explains the synthetic
panel, observable model inputs, objective/context ablation, demographic
allocation choices, bounds, and raw cross-report consistency checks. Re-run
the underlying experiment with:

```bash
PYTHONPATH=src .venv/bin/python -m reference_calibration.provider_benchmark \
  --profile full \
  --output-dir outputs/provider_model_final
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
.venv/bin/python -m reference_calibration.provider_benchmark \
  --profile full --output-dir outputs/provider_model_final
```

The first command reproduces the original measurement-layer experiment. The
second reproduces the provider-model and demographic-allocation comparison.
The full profiles take longer because they use more campaigns, users, report
shapes, and stress cases. The original experiment writes:

- `summary.json`: machine-readable configuration and headline metrics;
- `model_artifacts.json`: fitted parameters and version metadata for both
  calibration options;
- `metrics.csv`: detailed accuracy and consistency measurements;
- `validation_report.md`: readable findings and acceptance checks;
- `error_by_scenario.png`: baseline and calibrated error comparison.
- `linkage_shift_sweep.png`: break-even behavior as audience matchability shifts.

The provider benchmark writes `provider_summary.json`, `provider_metrics.csv`,
and total-reach and demographic-error charts to `outputs/provider_model_final/`.

## What is validated

The original calibration experiment fits on large, broadly targeted campaigns.
The provider experiment instead trains on a diverse synthetic panel containing
all tested campaign objectives and audience strategies. In both cases, entire
campaigns are assigned to fitting, holdout, or independent evaluation splits.
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
