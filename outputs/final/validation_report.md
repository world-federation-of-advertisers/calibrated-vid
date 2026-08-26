# Synthetic validation report

## Scope

- Population represented: 120,000,000 people using 30,000 weighted synthetic people.
- EDPs: 10; weeks: 13; Reference-ID pool: 5,000,000,000.
- Calibration campaigns: 24 fitting and 8 whole-campaign holdouts.
- Stress campaigns per scenario: 6.
- Conditional email agreement ranges from approximately 52% to 72%; email availability ranges from 10% to 95%.

## Models

- Selected fixed/log candidate: `pair_aware_fixed_plus_shared_log` (54 fitted parameters).
- Alternative: `two_group_latent_mixture` (21 fitted parameters at 10 EDPs).
- The pair-aware family was selected only among its fixed, shared-log, and order-log submodels using whole-campaign holdouts.

## Headline nominal-stress accuracy

| Method | Mean absolute relative error | p90 | p99 | Maximum |
|---|---:|---:|---:|---:|
| Existing VID baseline | 18.55% | 56.37% | 190.32% | 193.42% |
| Pair-aware fixed/log | 11.20% | 36.93% | 116.95% | 119.04% |
| Two-group mixture | 9.82% | 27.61% | 111.43% | 113.54% |

These figures combine the explicit 2-, 5-, and 10-EDP report shapes across representative, small-versus-large, two-small, all-small, and mixed-objective scenarios. Deliberate matchability-transfer shifts are reported separately below. Relative error can exceed 100% when the true union is small.

### Mean error by nominal scenario

| Scenario | Existing VID | Pair-aware fixed/log | Two-group mixture |
|---|---:|---:|---:|
| all small correlated | 83.18% | 50.50% | 44.07% |
| mixed objectives | 7.68% | 3.79% | 4.50% |
| representative | 0.37% | 2.16% | 2.36% |
| small vs large nonreach | 3.36% | 5.30% | 3.40% |
| two small correlated | 15.74% | 4.47% | 3.57% |
| two small disjoint | 0.96% | 0.99% | 1.02% |

### Mean error by report size

| EDPs | Existing VID | Pair-aware fixed/log | Two-group mixture |
|---:|---:|---:|---:|
| 2 | 9.75% | 4.21% | 2.34% |
| 5 | 14.40% | 9.40% | 8.20% |
| 10 | 35.65% | 21.80% | 20.54% |

### Requested stress examples

| Campaign/report shape | Existing VID | Pair-aware fixed/log | Two-group mixture |
|---|---:|---:|---:|
| Small non-reach versus large reach, weeks 5-12, 2 EDPs | 6.29% | 2.09% | 0.52% |
| Two small correlated campaigns, full flight, 2 EDPs | 49.14% | 16.07% | 4.55% |
| Two small disjoint campaigns, full flight, 2 EDPs | 2.57% | 2.57% | 2.57% |
| All-small correlated, weeks 7-13, 5 EDPs | 77.99% | 44.82% | 36.83% |
| Mixed objectives, full flight, 10 EDPs | 15.08% | 4.17% | 5.87% |

## Calibration-transfer sweep

The sweep changes how strongly campaign selection favors people who are generally easy or difficult to match. Zero adds no direct matchability selection; negative values select less-matchable people and positive values select more-matchable people.

| Shift | Existing VID | Pair-aware fixed/log | Two-group mixture |
|---:|---:|---:|---:|
| -1.00 | 96.60% | 82.05% | 86.67% |
| -0.75 | 59.99% | 47.98% | 52.77% |
| -0.50 | 32.85% | 22.35% | 26.99% |
| -0.25 | 16.43% | 8.13% | 13.13% |
| +0.00 | 9.75% | 4.58% | 7.35% |
| +0.25 | 15.54% | 8.98% | 2.18% |
| +0.50 | 32.11% | 12.22% | 9.51% |
| +0.75 | 59.62% | 20.98% | 21.53% |
| +1.00 | 95.46% | 41.47% | 40.40% |

A simple observable diagnostic—the average pairwise log difference between observed and expected Reference-ID capture—does respond to the injected shift, but it is also affected by genuine campaign-overlap differences:

| Model | Correlation with injected shift | Mean absolute score on nominal stress campaigns |
|---|---:|---:|
| Pair-aware fixed/log | 0.87 | 1.616 |
| Two-group mixture | 0.88 | 1.609 |

The diagnostic is therefore useful for review prioritization, not as a reliable automatic test that linkage has shifted.

## Consistency and failure handling

- Exact-repeat failures: 0.
- Monotonic violations remaining across stored reports: 0.
- Set-coverage inequality violations remaining where all required reports existed: 0.
- Deliberately infeasible fixture produced a result: True; status: `REVIEW_REQUIRED`; prior results unchanged: True.

Request order can matter in principle because earlier finalized reports become immutable anchors. It did not change any result in the tested request set:

| Model | Mean relative range | p90 | Maximum |
|---|---:|---:|---:|
| pair_aware_fixed_plus_shared_log | 0.00% | 0.00% | 0.00% |
| two_group_latent_mixture | 0.00% | 0.00% | 0.00% |

## Interpretation

Passing the structural tests shows that the implementation can produce bounded reports, preserve finalized history, recognize exact repeats, and flag infeasible new requests. It does not prove that the calibration assumptions transfer to real campaigns.

The most important next validation is to repeat the model fitting and holdout comparison on approved aggregate observations from real broad-reach and non-reach campaigns. In particular, real data is needed to determine whether pair-specific affinity, campaign-size effects, or latent person-level matchability best explains Reference-ID visibility.

## Acceptance checks

- [x] Exact repeated requests are stable.
- [x] An infeasible new report is still produced and flagged.
- [x] Infeasible processing does not rewrite prior results.
- [x] Ten-EDP reports are included.
- [x] Both calibration families are evaluated.
- [x] Selected pair-aware model has representative holdout p90 union error at or below 10% (6.86%).
- [x] Pair-aware calibration improves mean nominal-stress error over the existing VID baseline.
- [x] Mixture calibration improves mean nominal-stress error over the existing VID baseline.

## Reproducibility artifacts

The fitted coefficients, mixture parameters, and version identifiers are written to `model_artifacts.json`. Detailed observations are in `metrics.csv`, and the full run configuration and acceptance results are in `summary.json`.
