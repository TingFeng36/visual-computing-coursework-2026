import unittest
from training_safety import apply_checkpoint_safety


class CheckpointRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        apply_checkpoint_safety()

    def test_zero_ap_first_epoch_and_later_comparisons(self):
        from rfdetr.util.utils import BestMetricSingle
        metric = BestMetricSingle()
        self.assertTrue(metric.update(0.0, 0))
        self.assertEqual(metric.best_ep, 0)
        self.assertFalse(metric.update(0.0, 1))
        self.assertTrue(metric.update(.2, 2))
        self.assertFalse(metric.update(.1, 3))

    def test_zero_ap_ema_holder_requests_regular_checkpoint(self):
        from rfdetr.util.utils import BestMetricHolder
        metric = BestMetricHolder(use_ema=True)
        self.assertTrue(metric.update(0.0, 0, is_ema=False))
        self.assertFalse(metric.update(0.0, 0, is_ema=True))
        self.assertTrue(metric.update(.1, 1, is_ema=True))


if __name__ == '__main__':
    unittest.main()
