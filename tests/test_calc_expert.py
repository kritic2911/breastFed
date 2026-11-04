"""Unit tests for high-resolution calcification expert and MIL pooling."""

import torch

from src.models.calc_expert import CalcificationExpertMIL, PatchConvNet


def test_patch_convnet_forward():
    net = PatchConvNet(in_channels=1, embed_dim=128)
    net.eval()
    x = torch.randn(4, 1, 256, 256)
    with torch.no_grad():
        logits, embed = net(x)
    assert logits.shape == (4, 1)
    assert embed.shape == (4, 128)


def test_calc_expert_mil_pooling_modes():
    # Test max, top_k, and attention pooling
    for mode in ["max", "top_k", "attention"]:
        expert = CalcificationExpertMIL(pooling_mode=mode)
        expert.eval()

        # Non-empty bag (3 patches)
        bag = torch.randn(3, 1, 256, 256)
        with torch.no_grad():
            logit, patch_scores = expert.forward_bag(bag)
        assert logit.ndim == 0 or logit.numel() == 1
        assert patch_scores is not None
        assert len(patch_scores) == 3

        # Empty bag (0 patches) -> returns default bias
        with torch.no_grad():
            empty_logit, empty_patch_scores = expert.forward_bag(None)
        assert empty_logit.ndim == 0 or empty_logit.numel() == 1
        assert empty_patch_scores is None
