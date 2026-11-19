"""Independent 9K-parameter pixel-prompt research decoder (not BlackVIP)."""
import torch
import torch.nn.functional as F

CHANNELS = 29
TRIGGER_SIZE = 3 * 16 * 16
PARAMS = CHANNELS*2*9 + CHANNELS + CHANNELS*CHANNELS*9 + CHANNELS + 3*CHANNELS + 3 + TRIGGER_SIZE
assert PARAMS == 9007


def initial_parameters(seed):
    g = torch.Generator().manual_seed(seed + 31)
    w1 = torch.randn(CHANNELS*2*9, generator=g, dtype=torch.float64) * (2/(2*9))**.5
    b1 = torch.zeros(CHANNELS, dtype=torch.float64)
    w2 = torch.randn(CHANNELS*CHANNELS*9, generator=g, dtype=torch.float64) * (2/(CHANNELS*9))**.5
    b2 = torch.zeros(CHANNELS, dtype=torch.float64)
    w3 = torch.randn(3*CHANNELS, generator=g, dtype=torch.float64) * .01
    b3 = torch.zeros(3, dtype=torch.float64)
    trigger = torch.zeros(TRIGGER_SIZE, dtype=torch.float64)
    return torch.cat((w1, b1, w2, b2, w3, b3, trigger))


def image_prompt(feature, theta, shape):
    if feature.ndim != 2 or feature.shape[1] != 32 or theta.numel() != PARAMS or tuple(shape) != (3,64,64):
        raise ValueError('expected features [batch,32], 9007 parameters and 3x64x64 images')
    n = 0
    def take(k):
        nonlocal n
        result = theta[n:n+k]; n += k
        return result
    w1 = take(CHANNELS*2*9).reshape(CHANNELS,2,3,3); b1 = take(CHANNELS)
    w2 = take(CHANNELS*CHANNELS*9).reshape(CHANNELS,CHANNELS,3,3); b2 = take(CHANNELS)
    w3 = take(3*CHANNELS).reshape(3,CHANNELS,1,1); b3 = take(3)
    trigger = take(TRIGGER_SIZE).reshape(1,3,16,16)
    x = feature.reshape(-1,2,4,4)
    x = F.gelu(F.conv2d(x,w1,b1,padding=1))
    x = F.gelu(F.conv2d(x,w2,b2,padding=1))
    x = F.interpolate(x,size=(16,16),mode='nearest')
    x = F.conv2d(x,w3,b3) + trigger
    return F.interpolate(x,size=(64,64),mode='nearest')
