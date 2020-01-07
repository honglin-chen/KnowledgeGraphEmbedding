import utils.math_utils as math
import torch

MIN_NORM = 1e-15


def expmap(v, c):
    """ Defining all the parameters in the Euclidean tangent space at the origin """
    v_norm = v.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    prod = c ** 0.5 * v_norm
    y = math.tanh(prod) * v / prod.clamp_min(MIN_NORM)
    origin = torch.zeros_like(y)
    return mobius_add(origin, y, c)


def mobius_add(x, y, c, dim=-1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def dist(x, y, c):
    xy = - mobius_add(x, y, c)
    norm = xy.norm(dim=-1, p=2, keepdim=True)
    c_sqrt = c ** 0.5
    return 2 * math.artanh(c_sqrt * norm) / c_sqrt