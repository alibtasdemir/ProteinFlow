import torch


def sample_prior(x, sigma, harmonic=False):
    if harmonic:
        ##TODO
        raise NotImplementedError
    else:
        prior = torch.randn_like(x)
        return prior * sigma

