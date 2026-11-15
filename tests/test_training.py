import torch

from clickseg.smoke import segmentation_loss


def test_loss_penalizes_missing_small_foreground_and_has_finite_gradient():
    target = torch.zeros((1, 1, 8, 8))
    target[:, :, 2:4, 2:4] = 1
    good = torch.where(target.bool(), 8.0, -8.0).requires_grad_()
    missing = torch.full_like(target, -8.0).requires_grad_()
    assert segmentation_loss(good, target) < segmentation_loss(missing, target)
    loss = segmentation_loss(missing, target)
    loss.backward()
    assert torch.isfinite(missing.grad).all()
    assert (missing.grad[:, :, 2:4, 2:4] < 0).all()
