import torch
from torch.func import functional_call, jacrev, vmap


def empirical_ntk(network, x, gamma=1., batch_size=16):
    """Trace over output coordinates, accumulated one parameter tensor at a time.

    Uses the uncentered trainable branch derivative / gamma. A frozen initial
    branch has zero parameter derivative. Returns the output-trace N x N NTK,
    not the full N x N x C x C tensor.
    """
    params = dict(network.named_parameters())
    buffers = dict(network.named_buffers())
    kernel = torch.zeros((len(x), len(x)), dtype=x.dtype, device=x.device)
    for name, parameter in params.items():
        def f(p, sample):
            return functional_call(network, ({**params, name: p}, buffers), (sample,))/gamma
        j = torch.cat([vmap(jacrev(f), in_dims=(None, 0))(parameter, chunk).detach().flatten(2)
                       for chunk in x.split(batch_size)])
        kernel += torch.einsum("ncp,mcp->nm", j, j)
        del j
    return kernel.detach().cpu().numpy()
