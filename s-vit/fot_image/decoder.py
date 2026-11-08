"""Secret-compatible CNN mapping owner-local features to an image-space prompt."""
import torch
import torch.nn.functional as F


INPUT_CHANNELS = 2
HIDDEN_CHANNELS = 4
GRID = 4
FEATURE_DIM = INPUT_CHANNELS * GRID * GRID
PARAMS = HIDDEN_CHANNELS*INPUT_CHANNELS*3*3 + HIDDEN_CHANNELS + 3*HIDDEN_CHANNELS + 3


def initial_parameters(seed=20261107):
    generator = torch.Generator().manual_seed(seed + 31)
    # Nonzero initial convolution weights avoid the dead-gradient problem of an
    # entirely zero two-layer network. The seed and initialization are public.
    w1 = torch.randn(HIDDEN_CHANNELS*INPUT_CHANNELS*3*3, generator=generator, dtype=torch.float64)*.05
    b1 = torch.zeros(HIDDEN_CHANNELS, dtype=torch.float64)
    w2 = torch.randn(3*HIDDEN_CHANNELS, generator=generator, dtype=torch.float64)*.05
    b2 = torch.zeros(3, dtype=torch.float64)
    return torch.cat((w1, b1, w2, b2))


def rademacher(step, seed=20261107):
    g = torch.Generator().manual_seed(seed + 1000003*step)
    return 2*torch.randint(0, 2, (PARAMS,), generator=g, dtype=torch.int64).double()-1


def nearest_upsample(x, height, width):
    batch, channels, h, w = x.shape
    if height % h or width % w:
        raise ValueError('image dimensions must be divisible by decoder grid')
    # Pure reshape and expansion, shared by plaintext and MPC implementations.
    return x.reshape(batch, channels, h, 1, w, 1).expand(
        batch, channels, h, height//h, w, width//w
    ).reshape(batch, channels, height, width)


def image_prompt(feature, theta, ops, image_shape):
    if tuple(feature.shape) != (1, FEATURE_DIM):
        raise ValueError('decoder requires a single feature vector')
    n1 = HIDDEN_CHANNELS*INPUT_CHANNELS*3*3
    w1 = theta[:n1].reshape(HIDDEN_CHANNELS, INPUT_CHANNELS, 3, 3)
    b1 = theta[n1:n1+HIDDEN_CHANNELS].reshape(1, HIDDEN_CHANNELS, 1, 1)
    n2 = n1+HIDDEN_CHANNELS
    w2 = theta[n2:n2+3*HIDDEN_CHANNELS].reshape(3, HIDDEN_CHANNELS, 1, 1)
    b2 = theta[n2+3*HIDDEN_CHANNELS:].reshape(1, 3, 1, 1)
    x = feature.reshape(1, INPUT_CHANNELS, GRID, GRID)
    x = ops.conv(x, w1, padding=1) + b1
    x = ops.activation(x)
    x = nearest_upsample(x, image_shape[-2], image_shape[-1])
    return ops.conv(x, w2) + b2


class PlainDecoderOps:
    def conv(self, x, weight, padding=0):
        return F.conv2d(x, weight, padding=padding)

    def activation(self, x):
        from srelu.protocol import SReLUProfile
        if not torch.isfinite(x).all() or x.abs().max() >= 16:
            raise ArithmeticError('decoder sReLU input outside [-16, 16]')
        return SReLUProfile().fixed_point_corrected_reference(x).to(x.device)


class SecretDecoderOps:
    def __init__(self, backbone_ops):
        self.ops = backbone_ops

    def conv(self, x, weight, padding=0):
        return x.conv2d(weight, padding=padding)

    def activation(self, x):
        from srelu.protocol import prepare_srelu, ruler_srelu
        from training.secure import raw_share, from_share
        self.ops.check(x, 16 - 2**-17, 'decoder.srelu_input')
        with self.ops.span('ring3_prepare', 'decoder.srelu'):
            material = prepare_srelu(self.ops.base, self.ops.wide, tuple(x.shape))
        return from_share(ruler_srelu(self.ops.base, self.ops.wide, raw_share(x), material), 17)


def perturb_image(images, prompt, scale, fixture=False):
    """Add a pixel-space perturbation to an already normalized input.

    The real dataset uses per-channel normalization; the public fixture does not.
    No clipping is applied in either path.
    """
    if fixture:
        return images + prompt * scale
    std = images.new_tensor([.272, .265, .274]).reshape(1, 3, 1, 1) if isinstance(images, torch.Tensor) else None
    if std is None:
        # A public tensor on the same device; no private values are reconstructed.
        std = torch.tensor([.272, .265, .274], device=prompt.device, dtype=torch.float64).reshape(1, 3, 1, 1)
    return images + prompt * (scale / std)
