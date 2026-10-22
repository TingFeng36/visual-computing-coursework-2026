"""Real official Nano checkpoint/head mapping, CUDA update, save/reload check."""
import argparse
from pathlib import Path
import tempfile

from category_initialization import initialize_category_rows
from coco_tools import category_names
from coco_voc20 import VOC20_IDS
from common import build_nano, ROOT, write_json


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--output', default='verification/voc20_head_fix.json')
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError('steps must be positive')
    weights = ROOT/'weights/rf-detr-nano.pth'
    detector = build_nano(weights, 20, device=args.device, amp=False, checkpointing=True)
    net = detector.model.model
    report = initialize_category_rows(detector, weights, list(VOC20_IDS), [category_names()[i] for i in VOC20_IDS])
    heads = [net.class_embed, *net.transformer.enc_out_class_embed]
    assert len(heads) == 14
    assert all(tuple(h.weight.shape) == (21, 256) and h.out_features == 21 for h in heads)
    assert all(tuple(h.bias.shape) == (21,) for h in heads)
    losses = []
    reload_passed = False
    if args.device == 'cuda':
        from rfdetr.models import build_criterion_and_postprocessors
        from rfdetr.util.get_param_dicts import get_param_dict
        criterion, _ = build_criterion_and_postprocessors(detector.model.args)
        optimizer = torch.optim.AdamW(get_param_dict(detector.model.args, net), lr=1e-5)
        net.train()
        criterion.train()
        image = torch.rand(1, 3, 384, 384, device='cuda')
        targets = [{'labels': torch.tensor([1, 20], device='cuda'),
                    'boxes': torch.tensor([[.3, .4, .2, .3], [.7, .6, .15, .2]], device='cuda')}]
        initial = net.class_embed.weight.detach().clone()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            result = net(image, targets)
            assert result['pred_logits'].shape == (1, 3900, 21)
            loss_dict = criterion(result, targets)
            loss = sum(loss_dict[key]*criterion.weight_dict[key] for key in loss_dict if key in criterion.weight_dict)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(h.weight.grad is not None and torch.isfinite(h.weight.grad).all() for h in heads)
            torch.nn.utils.clip_grad_norm_(net.parameters(), .1)
            optimizer.step()
            losses.append(float(loss.detach()))
            print(f'VOC20 real Nano step {step+1}: loss={losses[-1]:.6f}', flush=True)
        assert not torch.equal(initial, net.class_embed.weight)
        net.eval()
        with torch.no_grad():
            before = net(image)['pred_logits'].detach().cpu()
        temporary_root = ROOT/'verification/test_tmp'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            checkpoint = Path(temporary)/'voc20.pth'
            torch.save({'model': net.state_dict()}, checkpoint)
            restored = build_nano(checkpoint, 20, device='cuda', amp=False, checkpointing=False)
            restored.model.model.eval()
            with torch.no_grad():
                after = restored.model.model(image)['pred_logits'].detach().cpu()
            assert before.shape == (1, 300, 21)
            assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)
            reload_passed = True
    result = {'passed': True, 'device': args.device, 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        'classifier_heads': len(heads), 'classifier_shape': [21, 256],
        'initialization': report, 'losses': losses, 'checkpoint_reload_passed': reload_passed,
        'scope': 'real official Nano and synthetic tensors; no real-data accuracy claim'}
    write_json(args.output, result)
    print('VOC20 classifier verification passed:', args.output, flush=True)


if __name__ == '__main__':
    main()
