"""Owner-local features and a small secret-compatible conditioned prompt decoder.

This is a low-rank conditional decoder, not a reproduction of BlackVIP's CNN.
Its linear layer is equivalent to a 1x1 convolution on a 1x1 feature map.
"""
import torch
import torch.nn.functional as F


class LocalEncoder:
    def __init__(self, mode, dim, device, mae_path=None, seed=104):
        self.mode, self.dim, self.device = mode, dim, device
        if mode == 'mae':
            if not mae_path:
                raise ValueError('--mae-path must point to local pretrained ViT-MAE weights')
            from transformers import ViTMAEModel
            self.model = ViTMAEModel.from_pretrained(mae_path, local_files_only=True).eval().to(device)
            self.model.config.mask_ratio = 0.0
            self.input_dim = self.model.config.hidden_size
        elif mode == 'fixture':
            self.input_dim = 3
        else:
            raise ValueError('unknown encoder')
        g = torch.Generator().manual_seed(seed)
        self.projection = (torch.randn(self.input_dim, dim, generator=g, dtype=torch.float64)
                           / self.input_dim ** .5).to(device)

    @torch.no_grad()
    def __call__(self, images):
        images = images.to(self.device)
        if self.mode == 'mae':
            # Undo the ViTLite input normalization, then use MAE's standard ImageNet normalization.
            mean = images.new_tensor([.480, .448, .397]).view(1, 3, 1, 1)
            std = images.new_tensor([.272, .265, .274]).view(1, 3, 1, 1)
            pixels = (images * std + mean).clamp(0, 1).float()
            pixels = F.interpolate(pixels, size=(224, 224), mode='bicubic', align_corners=False)
            pixels = (pixels - pixels.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)) / pixels.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
            # A fixed noise order prevents MAE's random token shuffle from changing +/- queries.
            n = (self.model.config.image_size // self.model.config.patch_size) ** 2
            noise = torch.arange(n, device=self.device, dtype=pixels.dtype).expand(pixels.shape[0], -1)
            feature = self.model(pixel_values=pixels, noise=noise).last_hidden_state[:, 0].double()
        else:
            feature = images.double().mean(dim=(-2, -1))
        return torch.tanh(feature @ self.projection)


def conditioned_latent(theta, feature, rank, feature_dim):
    """theta contains rank biases followed by rank*feature_dim shared weights."""
    if feature.shape[0] != 1:
        raise ValueError('prototype currently supports batch size 1')
    return theta[:rank] + theta[rank:].reshape(rank, feature_dim).matmul(feature.reshape(feature_dim, 1)).reshape(rank)


def direction(step, dim, seed):
    generator = torch.Generator().manual_seed(seed + 1000003 * step)
    u = torch.randn(dim, generator=generator, dtype=torch.float64)
    return u / u.norm()
