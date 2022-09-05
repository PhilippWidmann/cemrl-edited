import torch
import torch.nn.functional as F


def product_of_gaussians3D(mus, sigmas_squared, padding_mask=None):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    inverse_sigmas_squared = torch.reciprocal(sigmas_squared)

    if padding_mask is not None:
        data_mask = torch.tensor(~padding_mask, device=mus.device).unsqueeze(2)
        inverse_sigmas_squared = data_mask * inverse_sigmas_squared
        mus = data_mask * mus

    sigma_squared = 1. / torch.sum(inverse_sigmas_squared, dim=1)
    mu = sigma_squared * torch.sum(mus * inverse_sigmas_squared, dim=1)
    return mu, sigma_squared


def process_gaussian_parameters(mu_sigma, latent_dim, sigma_ops="softplus", mode=None, padding_mask=None):
    """
    Generate a Gaussian distribution given a selected parametrization.
    """
    mus, sigmas = torch.split(mu_sigma, split_size_or_sections=latent_dim, dim=-1)

    if sigma_ops == 'softplus':
        # Softplus, s.t. sigma is always positive
        # sigma is assumed to be st. dev. not variance
        sigmas = F.softplus(sigmas)
    if mode == 'multiplication':
        mu, sigma = product_of_gaussians3D(mus, sigmas, padding_mask=padding_mask)
    else:
        mu = mus
        sigma = sigmas
    return torch.cat([mu, sigma], dim=-1)
