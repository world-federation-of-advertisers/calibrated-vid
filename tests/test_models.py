import unittest

import numpy as np

from reference_calibration.measurement import CalibrationDataset
from reference_calibration.experiment import _select_pair_model
from reference_calibration.models import (
    LatentMixtureModel,
    PairAwareLogModel,
    _balance_campaign_weights,
    calibration_model_from_dict,
)
from reference_calibration.research_models import (
    DirectPairLogModel,
    HierarchicalPairModel,
    LowRankAffinityModel,
    MonotoneSplineCaptureModel,
    MultiGroupMixtureModel,
)


def synthetic_dataset(n_edps: int = 10) -> CalibrationDataset:
    masks = np.array([mask for mask in range(1, 1 << n_edps) if mask.bit_count() >= 2], dtype=np.int64)
    orders = np.array([int(mask).bit_count() for mask in masks], dtype=np.int8)
    log_scale = np.full(len(masks), np.log(0.45))
    q = np.linspace(0.60, 0.15, n_edps)
    capture = np.array([np.prod(q[[i for i in range(n_edps) if mask & (1 << i)]]) for mask in masks])
    k0 = np.full(len(masks), 2_000_000.0)
    return CalibrationDataset(
        campaign_ids=np.array(["c0"] * len(masks), dtype=object),
        subset_masks=masks,
        subset_orders=orders,
        log_scale=log_scale,
        k0=k0,
        signal=k0 * capture,
        truth=k0,
        weight=np.sqrt(k0),
    )


class CalibrationModelTest(unittest.TestCase):
    def test_campaign_weight_balancing(self):
        campaign_ids = np.array(["a", "a", "b", "b"], dtype=object)
        balanced = _balance_campaign_weights(
            campaign_ids,
            np.array([100.0, 100.0, 1.0, 1.0]),
        )
        self.assertAlmostEqual(
            np.linalg.norm(balanced[campaign_ids == "a"]),
            np.linalg.norm(balanced[campaign_ids == "b"]),
        )

    def test_pair_parameter_counts(self):
        data = synthetic_dataset()
        expected = {"none": 53, "shared": 54, "by_order": 62}
        for mode, count in expected.items():
            model = PairAwareLogModel.fit(data, 10, mode, 0.1)
            self.assertEqual(model.parameter_count, count)
            prediction = model.predict_capture(data.subset_masks, data.log_scale)
            self.assertTrue(np.all((prediction > 0) & (prediction < 1)))

    def test_mixture_is_bounded(self):
        data = synthetic_dataset()
        model = LatentMixtureModel.fit(data, 10, 123)
        prediction = model.predict_capture(data.subset_masks, data.log_scale)
        self.assertEqual(model.parameter_count, 21)
        self.assertTrue(np.all((prediction > 0) & (prediction <= 1)))

    def test_model_artifacts_round_trip(self):
        data = synthetic_dataset()
        for model in (
            PairAwareLogModel.fit(data, 10, "shared", 0.1),
            LatentMixtureModel.fit(data, 10, 123),
        ):
            restored = calibration_model_from_dict(model.describe())
            np.testing.assert_allclose(
                restored.predict_capture(data.subset_masks, data.log_scale),
                model.predict_capture(data.subset_masks, data.log_scale),
            )

    def test_pair_selection_requires_material_improvement(self):
        candidates = [
            PairAwareLogModel(10, "none", np.zeros(53), 0.1, "fixed"),
            PairAwareLogModel(10, "shared", np.zeros(54), 0.1, "shared"),
            PairAwareLogModel(10, "by_order", np.zeros(62), 0.1, "order"),
        ]
        rows = []
        for name, values in {
            "fixed": [0.08, 0.09, 0.10],
            "shared": [0.07, 0.08, 0.09],
            "order": [0.069, 0.079, 0.089],
        }.items():
            rows.extend(
                {
                    "category": "holdout_union",
                    "model": name,
                    "value": value,
                }
                for value in values
            )
        selected = _select_pair_model(candidates, rows, material_improvement=0.005)
        self.assertEqual(selected.name, "shared")

    def test_research_models_are_bounded(self):
        data = synthetic_dataset()
        pair_rows = data.subset_orders == 2
        direct = DirectPairLogModel.fit(data, 10)
        direct_prediction = direct.predict_capture(
            data.subset_masks[pair_rows],
            data.log_scale[pair_rows],
        )
        self.assertEqual(direct.parameter_count, 46)
        self.assertTrue(np.all((direct_prediction > 0) & (direct_prediction < 1)))

        models = (
            HierarchicalPairModel.fit(data, 10),
            MonotoneSplineCaptureModel.fit(data, 10),
            LowRankAffinityModel.fit(data, 10, rank=2, seed=123),
            MultiGroupMixtureModel.fit(data, 10, n_groups=3, seed=123),
        )
        for model in models:
            prediction = model.predict_capture(data.subset_masks, data.log_scale)
            self.assertTrue(np.all((prediction > 0) & (prediction <= 1)), model.name)


if __name__ == "__main__":
    unittest.main()
