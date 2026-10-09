import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from runtime import build_model, load_config, loss, PairedDataset, prediction, preprocess, split_check

torch.set_num_threads(2)


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = load_config()

    def test_forward_backward_and_batch_independence(self):
        cfg = self.config
        x = preprocess(np.full(cfg['feature_shape'], .4 * cfg['feature_scale'], np.float32), cfg)
        x = torch.stack([x, x * .6, x * .8])
        net = build_model(cfg)
        pred, coef = prediction(net, x)
        self.assertEqual(pred.shape, (3, cfg['output_dim']))
        objective = loss(pred, torch.full_like(pred, .2), coef, cfg)
        objective.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        net.eval()
        with torch.no_grad():
            batch, _ = prediction(net, x)
            single, _ = prediction(net, x[:1])
        torch.testing.assert_close(batch[:1], single, atol=1e-5, rtol=1e-4)

    def test_reject_invalid_feature_and_missing_target(self):
        cfg = self.config
        with self.assertRaises(ValueError):
            preprocess(np.zeros((1,)), cfg)
        with self.assertRaises(ValueError):
            preprocess(np.full(cfg['feature_shape'], np.nan), cfg)
        with tempfile.TemporaryDirectory() as temp:
            np.save(Path(temp) / 'rgb_a.npy', np.zeros(cfg['feature_shape']))
            with self.assertRaises(FileNotFoundError):
                PairedDataset(temp, cfg)

    def test_paired_data_and_split_overlap(self):
        cfg = self.config
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root / 'rgb_a.npy', np.zeros(cfg['feature_shape'], np.float32))
            np.save(root / 'spectral_a.npy', np.ones(cfg['output_dim'], np.float32))
            ds = PairedDataset(root, cfg)
            x, y = ds[0]
            self.assertEqual(y.shape, (cfg['output_dim'],))
            torch.testing.assert_close(x, preprocess(np.zeros(cfg['feature_shape']), cfg))
            with self.assertRaises(ValueError):
                split_check(ds, ds)

    def test_dc_and_tv_are_independent_terms(self):
        cfg = copy.deepcopy(self.config)
        cfg['loss'] = {'dc': .1, 'tv': .2, 'coef': .3}
        p = torch.tensor([[1., 2., 3.]])
        y = p + 1
        coef = torch.tensor([[2.]])
        self.assertAlmostEqual(loss(p, y, coef, cfg).item(), 1 + .1 + .2 + 1.2, places=5)

    def test_dual_basis_and_grid(self):
        if self.config['model_class'] != 'DualHeadSpectralNet':
            self.skipTest('Dual-head stage only')
        cfg = self.config
        net = build_model(cfg)
        basis = net.basis
        torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1]), atol=2e-6, rtol=1e-5)
        self.assertEqual(net.attn_weight.shape[1], cfg['model_kwargs']['token_grid'][0] * cfg['model_kwargs']['token_grid'][1])
        net.eval()
        x = preprocess(np.full(cfg['feature_shape'], .5 * cfg['feature_scale']), cfg).unsqueeze(0)
        with torch.no_grad():
            y, coef = prediction(net, x)
        self.assertTrue((y >= 0).all())
        self.assertEqual(coef.shape[1], basis.shape[1])


if __name__ == '__main__':
    unittest.main()
