# Campaign-specific calibrated VID demonstration

This repository contains one deliberately small synthetic experiment. It asks:

1. What happens when a VID model learns cross-publisher overlap from broad Reach campaigns?
2. Does that fixed overlap transfer to campaigns that reach a different latent population?
3. Can a campaign-specific shared-fingerprint signal adjust the shared virtual population without changing either publisher's single-publisher reach?

The experiment has two publishers and two virtual populations with identical expected single-publisher reaches. The broad Reach population has higher cross-publisher affinity than the response population used by Traffic campaigns. Ten Reach campaigns fit the models; ten unseen Reach campaigns and ten unseen Traffic campaigns evaluate them.

The fixed model learns the normal overlap rate from the Reach training campaigns. The fingerprint observation model separately specifies 30% coverage for publisher A, 80% coverage for publisher B, and 60% conditional agreement when the same person has fingerprints at both publishers. Therefore only 14.4% of true overlaps are expected to appear as direct fingerprint matches.

The calibrated model first corrects the raw match count for that observation probability and then adds one parameter:

```text
fingerprint overlap estimate = observed matches
  / (0.30 * 0.80 * 0.60 * smaller publisher reach)

campaign overlap = Reach baseline
  + reference sensitivity * (fingerprint overlap estimate - Reach baseline)
```

Both models preserve the observed single-publisher reaches. They differ only in the size of the shared virtual population.

The full person-level overlap remains hidden until evaluation, so the notebook does not grade the model against its held-out labels.

## Held-out result

| Campaign family | Fixed overlap MAE | Fingerprint-aware overlap MAE | Fixed total unique-reach MAPE | Fingerprint-aware total unique-reach MAPE |
| --- | ---: | ---: | ---: | ---: |
| Reach | 6.82 points | 0.73 points | 3.93% | 0.43% |
| Traffic | 30.48 points | 0.34 points | 14.85% | 0.16% |

For held-out Traffic campaigns, the fixed Reach baseline overstates overlap by approximately 160%. The primary planning consequence is a 38% understatement of the smaller publisher's incremental unique reach. The same absolute audience miss is a 15% error when divided by the much larger total cross-publisher unique reach.

The strong calibrated result is intentional: the simulator guarantees that fingerprint availability and agreement are independent of campaign exposure. Real deployment must test those assumptions independently.

## Files

- `proto/calibrated_vid/synthetic_campaign_experiment.proto`: standalone experiment and fitted-model schema, inspired by the [`ImpressionTestDataConfig`](https://github.com/world-federation-of-advertisers/cross-media-measurement/blob/main/src/main/proto/wfa/measurement/integration/k8s/testing/impression_test_data_config.proto) event-group shape without adding that repository as a dependency.
- `configs/two_population_experiment.textproto`: the two populations and 30 campaign realizations.
- `notebooks/two_population_calibration.ipynb`: executed explanation and results.
- `src/calibrated_vid/experiment.py`: deterministic simulation, fitting, and evaluation.
- `tests/test_experiment.py`: invariants and headline-result tests.

## Reproduce

```bash
./scripts/bootstrap.sh
.venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python notebooks/build_two_population_notebook.py
```

The experiment is feasibility evidence, not a production accuracy claim. Its key assumptions are that the fingerprint-bearing subset has the same overlap behavior as the rest of each campaign and that the coverage and agreement parameters are known. Independent panel validation is required before applying those assumptions to real campaigns.
