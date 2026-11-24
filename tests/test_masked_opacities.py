"""Masked rasterization (MaskGaussian) through negative opacities."""

import pytest
import torch

device = torch.device("cuda:0")
WIDTH, HEIGHT = 40, 32


def scene(count=24, seed=0):
    generator = torch.Generator().manual_seed(seed)
    means = torch.empty(count, 3)
    means[:, :2] = torch.rand(count, 2, generator=generator) * 2 - 1
    means[:, 2] = 3 + torch.rand(count, generator=generator) * 2
    quats = torch.randn(count, 4, generator=generator)
    scales = 0.05 + torch.rand(count, 3, generator=generator) * 0.2
    opacities = 0.2 + torch.rand(count, generator=generator) * 0.7
    colors = torch.rand(count, 3, generator=generator)
    masks = (torch.rand(count, generator=generator) > 0.4).float()
    viewmats = torch.eye(4)[None]
    Ks = torch.tensor([[[30.0, 0, WIDTH / 2], [0, 30.0, HEIGHT / 2], [0, 0, 1]]])
    return [t.to(device) for t in (means, quats, scales, opacities, colors, masks, viewmats, Ks)]


def render(means, quats, scales, opacities, colors, viewmats, Ks, **options):
    from gsplat import rasterization

    colors_out, _, meta = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, WIDTH, HEIGHT, packed=False, **options
    )
    return colors_out, meta


def reference(means2d, conics, depths, opacities, colors, masks):
    """Alpha blending in plain PyTorch, with each mask scaling alpha."""
    lines, columns = torch.meshgrid(
        torch.arange(HEIGHT, device=device), torch.arange(WIDTH, device=device), indexing="ij"
    )
    pixels = torch.stack([columns, lines], -1).float() + 0.5  # [H, W, 2]
    order = torch.argsort(depths)
    image = torch.zeros(HEIGHT, WIDTH, 3, device=device)
    transmittance = torch.ones(HEIGHT, WIDTH, device=device)
    for gaussian in order.tolist():
        delta = means2d[gaussian] - pixels
        a, b, c = conics[gaussian]
        sigma = 0.5 * (a * delta[..., 0] ** 2 + c * delta[..., 1] ** 2) + b * delta[..., 0] * delta[..., 1]
        vis = torch.exp(-sigma)
        valid = (sigma >= 0) & (opacities[gaussian] * vis >= 1 / 255)
        alpha = torch.where(valid, masks[gaussian] * torch.clamp(opacities[gaussian] * vis, max=0.99), 0)
        image = image + (alpha * transmittance)[..., None] * colors[gaussian]
        transmittance = transmittance * (1 - alpha)
    return image


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_masked_gaussians_render_nothing():
    from gsplat import masked_opacities

    means, quats, scales, opacities, colors, masks, viewmats, Ks = scene()
    kept = masks > 0.5

    masked, _ = render(means, quats, scales, masked_opacities(opacities, masks), colors, viewmats, Ks)
    removed, _ = render(
        means[kept], quats[kept], scales[kept], opacities[kept], colors[kept], viewmats, Ks
    )

    torch.testing.assert_close(masked, removed, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_masked_gaussians_get_the_mask_gradient_of_a_scaled_alpha():
    means, quats, scales, opacities, colors, masks, viewmats, Ks = scene()
    weights = torch.rand(HEIGHT, WIDTH, 3, device=device)
    _, meta = render(means, quats, scales, opacities, colors, viewmats, Ks)
    assert (meta["radii"] > 0).all(), "every Gaussian must be on screen for the comparison"

    opacities_cuda = opacities.clone().requires_grad_()
    masks_cuda = masks.clone().requires_grad_()
    image, _ = render(
        means, quats, scales, opacities_cuda, colors, viewmats, Ks, gaussian_masks=masks_cuda
    )
    (image[0] * weights).sum().backward()

    opacities_reference = opacities.clone().requires_grad_()
    masks_reference = masks.clone().requires_grad_()
    image = reference(
        meta["means2d"][0].detach(), meta["conics"][0].detach(), meta["depths"][0].detach(),
        opacities_reference, colors, masks_reference,
    )
    (image * weights).sum().backward()

    assert (masks_cuda.grad[masks < 0.5] != 0).any(), "masked Gaussians must get a gradient"
    torch.testing.assert_close(masks_cuda.grad, masks_reference.grad, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(opacities_cuda.grad, opacities_reference.grad, atol=1e-4, rtol=1e-3)
