"""Small behavior checks for masking, summary reuse, and metric aggregation."""
import unittest
import torch
from ctlite import CTLite
from ctlite.data import evaluate


class CTLiteTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)
        self.model = CTLite(channels=3, rank=1, width=4, rates=[0., 2., 8.])
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(torch.randn_like(parameter) * .1)
        self.x = torch.randn(2, 6, 3)
        self.mask = (torch.rand_like(self.x) > .3).float()
        self.t = torch.linspace(0., .6, 6)[None, :, None].expand(2, -1, -1)
        self.q = torch.tensor([.7, .9])[None, :, None].expand(2, -1, -1)

    def test_masked_values_cannot_change_predictions(self):
        altered = self.x.masked_fill(~self.mask.bool(), float('nan'))
        a = self.model(self.x, self.t, self.mask, self.q)
        b = self.model(altered, self.t, self.mask, self.q)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_summary_reuse_matches_full_prediction(self):
        full = self.model(self.x, self.t, self.mask, self.q)
        cached = self.model.decode(self.model.encode(self.x, self.t, self.mask), self.q)
        torch.testing.assert_close(full, cached, rtol=0, atol=0)

    def test_targets_cannot_change_predictions(self):
        a = self.model(self.x, self.t, self.mask, self.q, y=torch.zeros(2, 2, 3))
        b = self.model(self.x, self.t, self.mask, self.q, y=torch.full((2, 2, 3), 1e6))
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_empty_history_is_finite(self):
        prediction = self.model(self.x, self.t, torch.zeros_like(self.mask), self.q)
        self.assertTrue(torch.isfinite(prediction).all())

    def test_metrics_weight_observed_scalars(self):
        class FixedPrediction(torch.nn.Module):
            def forward(self, prediction, **unused):
                return prediction

        data = dict(prediction=torch.tensor([[[2., 99.]], [[1., 1.]], [[99., 99.]]]),
                    y=torch.zeros(3, 1, 2),
                    y_mask=torch.tensor([[[1., 0.]], [[1., 1.]], [[0., 0.]]]))
        for batch_size in (1, 2, 3):
            result = evaluate(FixedPrediction(), data, batch_size)
            self.assertEqual(result['points'], 3)
            self.assertEqual(result['mse'], 2.)
            self.assertEqual(result['mae'], 4/3)


if __name__ == '__main__':
    unittest.main()
