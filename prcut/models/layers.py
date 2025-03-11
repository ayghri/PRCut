from typing import List
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import Distribution
from torch.distributions import kl
from torch import Tensor


class DecoupledLinear(nn.Module):
    def __init__(
        self, repr_dims: List[int], num_clusters: int, weight_norm: bool = False
    ):
        """
        Simple class where input is a list of representation dimensions
        The output is of shape (b, num_clusters, num_repr)
        Each representation has its own linear projection layer
        """
        super().__init__()
        self.repr_dims = repr_dims
        self.num_clusters = num_clusters

        if weight_norm:
            weight_norm_fn = lambda l: nn.utils.parametrizations.weight_norm(l)
        else:
            weight_norm_fn = lambda l: l

        self.linears = []
        for i, d in enumerate(repr_dims):
            layer = weight_norm_fn(nn.Linear(d, num_clusters))
            self.linears.append(layer)
            self.add_module(f"encoder_linear_{i}", layer)

    def forward(self, feats_per_space: List[torch.Tensor]) -> torch.Tensor:
        y_per_space = []
        for feats, linear in zip(feats_per_space, self.linears):
            y_per_space.append(linear(feats))
        y_per_space = torch.softmax(torch.stack(y_per_space, dim=-1), dim=-2)
        return y_per_space


class CentersMVA(nn.Module):
    def __init__(self, input_dim, num_clusters, alpha=0.95):
        super(CentersMVA, self).__init__()
        self.alpha = alpha
        self.centers = nn.Parameter(
            data=torch.randn(num_clusters, input_dim), requires_grad=False
        )
        self.register_parameter(name="centers", param=self.centers)

    def update_centers(self, z, p):
        """
        p (B, num_clusters)
        x (B, input_dim)

        """
        self.centers.data = self.alpha * self.centers.data + (1 - self.alpha) * (
            (1 - p).t().mm(z) / p.size(0)
        )
        self.centers.data /= self.centers.data.norm(dim=1, keepdim=True)

    def forward(self, x):
        return x.mm(self.centers.t())


class FlatInstanceNorm(nn.Module):
    def forward(self, x):
        return x * x.norm(dim=1, keepdim=True).reciprocal()


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape: tuple):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "shape={}".format(self.shape)


class CholeskyLayer(nn.Module):

    def __init__(self, n_feats, alpha=0.1, eps=1e-6):
        super().__init__()
        self.n_feats = n_feats
        self.alpha = alpha
        self.eps = eps
        self.L = nn.Parameter(torch.eye(n_feats), requires_grad=False)
        self.S = nn.Parameter(torch.eye(n_feats), requires_grad=False)
        self.eps_eye = nn.Parameter(self.eps * torch.eye(n_feats), requires_grad=False)
        # self.eps_eye = self.eps * torch.eye(self.n_feats)
        # self.eps_eye = self.eps * torch.eye(self.n_feats)
        self.register_parameter(name="sigma", param=self.S)
        self.register_parameter(name="eps_eye", param=self.eps_eye)
        self.register_parameter(name="cholesky_inv", param=self.L)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.S.data = (1 - self.alpha) * self.S
                self.S.data = self.S + self.alpha * x.t().mm(x) / x.size(0)
                self.L.data = torch.linalg.cholesky(self.S + self.eps_eye).inverse().t()
        return x.mm(self.L)


class CholeskyOrthogonal(nn.Module):

    def __init__(self, n, alpha=0.9, eps=1e-6):
        super(CholeskyOrthogonal, self).__init__()
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.L = nn.Parameter(torch.eye(n + 1), requires_grad=False)
        self.S = nn.Parameter(torch.eye(n + 1), requires_grad=False)
        self.register_parameter(name="sigma", param=self.S)
        self.register_parameter(name="cholesky", param=self.L)

    def forward(self, x):
        z = torch.nn.functional.pad(x, (1, 0), value=1.0)
        if self.training:
            with torch.no_grad():
                self.S.data = (1 - self.alpha) * z.t().mm(z) / x.size(
                    0
                ) + self.alpha * self.S
                self.L.data = torch.inverse(
                    torch.linalg.cholesky(self.S)
                    + self.eps * torch.eye(self.n + 1).to(self.S)
                ).t()

        return z.mm(self.L)[:, 1:]


# def conv3x3(in_planes, out_planes, stride=1):
#     "3x3 convolution with padding"
#     return nn.Conv2d(
#         in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
#     )
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         # self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.GELU()
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         # self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.GELU()
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class DistributionOutput:

    def __init__(self, /, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def __repr__(self):
        # Tensor.__repr__ = short_torch_repr
        items = [f"{k}={repr(v)}" for k, v in self.__dict__.items()]
        # Tensor.__repr__ = default_torch_repr
        return "{}({})".format(type(self).__name__, ",\n ".join(items))


class ParameterizedOutput(DistributionOutput):
    distribution: Distribution
    samples: Tensor


class NormalLayer(nn.Module):

    def __init__(self, latent_dim, gaussian_dim=None):
        super().__init__()
        if gaussian_dim is None:
            gaussian_dim = latent_dim
        self.gaussian_dim = gaussian_dim
        self.mu_layer = nn.Linear(latent_dim, gaussian_dim)
        self.logvar_layer = nn.Linear(latent_dim, gaussian_dim)

    def posterior_distribution(self, encoding) -> Distribution:
        # projection = self.projection_layer(encoding)
        # slice on the last index
        # mu = projection[..., : self.gaussian_dim]
        # log_var = projection[..., self.gaussian_dim :]
        mu = self.mu_layer(encoding)
        log_var = self.logvar_layer(encoding)
        # scale = sigma, and var = sigma^2
        scale = torch.exp(0.5 * log_var)
        return Normal(loc=mu, scale=scale)

    def forward(self, encoding) -> ParameterizedOutput:
        # mu = self.mu_layer(encoding)
        # log_var = self.logvar_layer(encoding)
        # z = sampling.reparameterize(mu=mu, log_var=log_var)
        dist = self.posterior_distribution(encoding)
        return ParameterizedOutput(distribution=dist, samples=dist.rsample())

    def kl_divergence(self, encoding, prior: Distribution):
        dist = self.posterior_distribution(encoding)
        return kl.kl_divergence(dist, prior)


class NormalLayerWrapper(nn.Module):

    def __init__(self, encoder, latent_dim, gaussian_dim=None):
        super().__init__()
        if gaussian_dim is None:
            gaussian_dim = latent_dim
        self.gaussian_dim = gaussian_dim
        self.encoder = encoder
        self.normal_layer = NormalLayer(latent_dim, gaussian_dim)

    def posterior_distribution(self, encoding) -> Distribution:
        # projection = self.projection_layer(encoding)
        # slice on the last index
        # mu = projection[..., : self.gaussian_dim]
        # log_var = projection[..., self.gaussian_dim :]
        mu = self.mu_layer(encoding)
        log_var = self.logvar_layer(encoding)
        # scale = sigma, and var = sigma^2
        scale = torch.exp(0.5 * log_var)
        return Normal(loc=mu, scale=scale)

    def parameterize(self, x) -> ParameterizedOutput:
        return self.normal_layer(self.encoder(x))

    def forward(self, x):
        encoding = self.encoder(x)
        # mu = self.mu_layer(encoding)
        # log_var = self.logvar_layer(encoding)
        # z = sampling.reparameterize(mu=mu, log_var=log_var)
        dist = self.normal_layer.posterior_distribution(encoding)
        return dist.rsample()

    def kl_divergence(self, encoding, prior: Distribution):
        dist = self.posterior_distribution(encoding)
        return kl.kl_divergence(dist, prior)


class StickBreaking(nn.Module):
    def forward(self, betas):
        p_betas = torch.nn.functional.pad(betas, (0, 1), value=1.0)
        p_betas[:, 1:].mul_(torch.cumprod(1 - betas, dim=1))
        return p_betas


if __name__ == "__main__":
    # import numpy as np

    # a = np.random.uniform(0, 1, (3, 3))
    # b = np.cumprod(1 - a, axis=1)
    # c = np.append(a, np.ones((b.shape[0], 1)), 1)
    # c[:, 1:] = c[:, 1:] * np.cumprod(1 - a, axis=1)
    # np.append(b, np.ones((b.shape[0], 1)), 1)
    # c = np.append(np.cumprod(a, axis=1), np.ones((b.shape[0], 1)), 1)
    # c.sum(1)
    # np.cumprod(np.append(a, np.ones((b.shape[0], 1)), 1), axis=1)
    # b[:,:-1]*a[:,:-1]
    # class DecoupledStickBreaking(nn.Module):
    import torch

    stick_b = StickBreaking()
    betas = torch.rand(3, 3)
    print(stick_b(betas))
    print(stick_b(betas).sum(1))
