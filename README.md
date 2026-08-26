# Reference-ID calibration synthetic validation

This package is a reproducible synthetic implementation of the design in
**Reference-ID Calibration and Cross-Report Reconciliation**. It implements:

- a single common Reference-ID join key hashed into a 5-billion-value pool;
- EDP-specific proprietary fallback identifiers that match across EDPs only
  through random pool collisions;
- heterogeneous email availability and approximately 60% conditional email
  agreement, with stable pair-specific affinity;
- complete pairwise-through-N intersection measurement for as many as 10 EDPs;
- the pair-aware fixed-plus-log calibration family;
- the two-group latent-matchability calibration family;
- nonnegative exclusive-cell decoding for internally valid union reach;
- stored-result reconciliation for reports over different EDP and week sets;
- production of a bounded `REVIEW_REQUIRED` result when prior results cannot be
  reconciled exactly.

The simulation never links a VID to a Reference ID. Synthetic truth is retained
only by the test harness so accuracy can be measured.

See `VALIDATION.md` for the standalone problem statement, design explanation,
results, limitations, and recommendation.

For a product-grounded walkthrough, open
`notebooks/meta_campaign_scenarios.ipynb`. It compares the existing VID
baseline with both calibration families across plausible Meta campaign types:
traffic, engagement retargeting, leads, sales, website and customer-list
retargeting, catalog campaigns, lookalikes, Advantage+ audience expansion, app
activity, unrelated niches, and a mixed-funnel portfolio. The notebook is
already executed. Every campaign type is also tested with weaker/stronger
cross-EDP audience similarity and lower/higher Reference-ID matchability. It
writes its detailed outputs to
`outputs/meta_campaign_scenarios/`.

Launch the notebook locally with:

```bash
.venv/bin/jupyter lab notebooks/meta_campaign_scenarios.ipynb
```

The broader method comparison is in
`notebooks/calibration_method_benchmark.ipynb`. Its checked-in full run compares
18 candidates and diagnostics, including direct, logit, and mixture pairwise
calibration, maximum-entropy higher-order inference, hierarchical and low-rank
models, two- and three-group mixtures, joint decoders, and a Bayesian/MAP
benchmark. Re-run the underlying experiment with:

```bash
PYTHONPATH=src .venv/bin/python -m reference_calibration.method_benchmark \
  --profile full \
  --campaigns-per-scenario 3 \
  --output-dir outputs/method_benchmark_final
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
```

The full profile takes longer because it uses more campaigns, users, report
orders, and stress cases. Both profiles write:

- `summary.json`: machine-readable configuration and headline metrics;
- `model_artifacts.json`: fitted parameters and version metadata for both
  calibration options;
- `metrics.csv`: detailed accuracy and consistency measurements;
- `validation_report.md`: readable findings and acceptance checks;
- `error_by_scenario.png`: baseline and calibrated error comparison.
- `linkage_shift_sweep.png`: break-even behavior as audience matchability shifts.

## What is validated

Calibration campaigns are large, broadly targeted campaigns. Entire campaigns,
including all their cumulative checkpoints, are assigned to either fitting or
holdout data. Stress campaigns include:

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

## Important interpretation

This is a synthetic validation harness, not evidence that either model is
correct for production data. Its purpose is to verify the mathematics,
interfaces, scaling behavior, failure handling, and test methodology. Model
selection should ideally be repeated on approved aggregate observations from
real calibration campaigns.
