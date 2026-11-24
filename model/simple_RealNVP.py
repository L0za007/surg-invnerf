"""
Simplyfied version of the Real-valued Non-Volume Preserving (RealNVP) model. 
Similar version is used in Omnimotion model.
"""
import numpy as np
import torch
from .affine import AffineNet, apply_TM

class NVPSimplified(torch.nn.Module):
    def __init__(self, n_layers, feature_dims, hidden_size, proj_dims, code_proj_hidden_size=[], aabb=[[0, 0, 0, 1, 1, 1]],
        proj_type="simple", pe_freq=4, normalization=True, affine=False, activation=torch.nn.LeakyReLU):
        """ Simple RealNVP model.
        Args:
            n_layers (int): Number of coupling layers.
            feature_dims (int):                  Size of the input features.
            hidden_size (int or list of int):    Size of the hidden layers.
            proj_dims (int):                     Size of the projection layer.
            code_proj_hidden_size (list of int): Size of the hidden layers for the code projector.
            proj_type (str):                     Type of projection layer. 
                Options['identity', 'simple', 'fixed_positional_encoding', 'gaussianrff', 'gabor'].
            pe_freq (int):                       Frequency of the positional encoding.
            normalization (bool):                Whether to use batch normalization.
            affine (bool):                       Whether to use an affine transformation.
            activation (torch.nn.Module):        Activation function.
        """
        super().__init__()
        self.affine = affine

        # make layers
        input_dims = 3 # x, y, z
        normalization = torch.nn.BatchNorm1d if normalization else None
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        self.layers = torch.nn.ModuleList()
        self.code_projectors = torch.nn.ModuleList()
        self.layer_idx = list(range(n_layers))

        aabb = torch.tensor(aabb[0], dtype=torch.float32)
        # max_t = aabb[3:]-aabb[:3] # [max_x, max_y, max_z]
        # self.register_buffer('max_t', max_t)
        
        # Genreating masks for each layer so we can use different pair of coordinates for each layer 
        mask_selection = [torch.randperm(input_dims)]
        for i in range(1, n_layers//input_dims):
            selec = torch.randperm(input_dims)
            if mask_selection[i-1][-1] == selec[0]:
                selec = selec.roll(1)
            mask_selection.append(selec)
        mask_selection = torch.cat(mask_selection)
        masks = torch.ones((n_layers,input_dims))
        masks[torch.arange(n_layers), mask_selection] = 0

        for i in self.layer_idx:
            # Projection Net for the temporal features in each layer # It just process the features and leave the shame shape at the input and output
            if len(code_proj_hidden_size) == 0:
                code_proj_hidden_size = [feature_dims]
            self.code_projectors.append(MLP( feature_dims, feature_dims, code_proj_hidden_size, 
                bn=normalization, act=activation))

            # Building block of the Real NVP model
            mask = masks[i][None, None, None]
            map_st = MLP(proj_dims + feature_dims, 2, hidden_size, bn=normalization, act=activation)
            ## proj is the projection layer that uses an encoder to project the input to a higher dimension (example positional encoding)
            proj = get_projection_layer(proj_dims=proj_dims, type=proj_type, pe_freq=pe_freq)
            self.layers.append(CouplingLayer(map_st, proj, mask))

        if self.affine:
            # this mlp takes time and depth as input and produce an affine transformation for x and y
            # The original implementatio used wrong keys. Try the performance of the model with num_layers = 2
            self.affine_mlp = AffineNet(input_dim=2, hidden_dim=256, skip_layers=[], 
                use_pe=True, pe_dims=[1], pe_freq=pe_freq, output_dim=5)
            
    def affine_transformation(self, t, x, inverse=False):
        """ Apply an affine transformation to the input x.
        Args:
            t (torch.Tensor): Time. [n_imgs]
            x (torch.Tensor): Input tensor. [n_imgs, n_pts, n_samples, 3]
            inverse (bool):   Whether to apply the inverse transformation.
        Returns:
            torch.Tensor: Transformed tensor."""
        depth = x[..., -1]  # [n_imgs, n_pts, n_samples]
        net_in = torch.stack([t[..., None].repeat(1, *x.shape[1:3]), depth], dim=-1) # [n_imgs, n_pts, n_samples, 2]
        tm_params = self.affine_mlp(net_in)  # [n_imgs, n_pts, n_samples, 5]
        affine = self.affine_mlp.get_ATM_from_params(tm_params, inverse=inverse)  # [n_imgs, n_pts, n_samples, 3, 3]
        xy = x[..., :2] # [n_imgs, n_pts, n_samples, 2]
        xy = apply_TM(affine, xy) # [n_imgs, n_pts, n_samples, 2]
        x = torch.cat([xy, depth[..., None]], dim=-1) # [n_imgs, n_pts, n_samples, 3]
        return x

    def _expand_features(self, F, x):
        """ Expands features to match the input dimensions of x
        :param F: (B, nF)
        :param x: (B, N, K, C)
        :returns: (B, N, K, nF)"""
        _, N, K, _ = x.shape
        return F[:, None, None, :].expand(-1, N, K, -1)

    def forward(self, t, feat, x):
        y = x

        if self.affine:
            y = self.affine_transformation(t, y)

        for i in self.layer_idx:
            feat_i = self.code_projectors[i](feat)
            feat_i = self._expand_features(feat_i, y)
            l1 = self.layers[i]
            y, _ = l1.forward(feat_i, y)#, self.max_t)

        return y

    def inverse(self, t, feat, y):
        x = y

        for i in reversed(self.layer_idx):
            feat_i = self.code_projectors[i](feat)
            feat_i = self._expand_features(feat_i, x)
            l1 = self.layers[i]
            x, _ = l1.inverse(feat_i, x)#, self.max_t)

        if self.affine:
            x = self.affine_transformation(t, x, inverse=True)

        return x

class CouplingLayer(torch.nn.Module):
    def __init__(self, map_st, projection, mask):
        super().__init__()
        self.map_st = map_st
        self.projection = projection
        self.register_buffer("mask", mask)
        #self.mask = mask

    def forward(self, F, y):#, max_t):
        y1 = y * self.mask
        F_y1 = torch.cat([F, self.projection(y[..., self.mask.squeeze().bool()])], dim=-1)
        st = self.map_st(F_y1)
        s, t = torch.split(st, split_size_or_sections=1, dim=-1)
        s = torch.tanh(s/4) * 4 #torch.clamp(s, min=-8, max=8)
        #t = torch.tanh(t) * max_t[~self.mask.squeeze().bool()]
        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj

    def inverse(self, F, x):#, max_t):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x[..., self.mask.squeeze().bool()])], dim=-1)
        st = self.map_st(F_x1)
        s, t = torch.split(st, split_size_or_sections=1, dim=-1)
        s = torch.tanh(s/4) * 4 #torch.clamp(s, min=-8, max=8)
        #t = torch.tanh(t) * max_t[~self.mask.squeeze().bool()]
        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)

        return y, ldj


class MLP(torch.nn.Module):
    def __init__(self, c_in, c_out, c_hiddens, act=torch.nn.LeakyReLU, bn=torch.nn.BatchNorm1d):
        """ Base MLP for projecting temporal features."""
        super().__init__()
        layers = []
        d_in = c_in
        for d_out in c_hiddens:
            layers.append(torch.nn.Linear(d_in, d_out))
            if bn is not None:
                layers.append(bn(d_out))
            layers.append(act())
            d_in = d_out
        layers.append(torch.nn.Linear(d_in, c_out))
        self.mlp = torch.nn.Sequential(*layers)
        self.c_out = c_out

    def forward(self, x):
        # x: B,...,C_in
        input_shape = x.shape
        C = input_shape[-1]
        _x = x.reshape(-1, C)  # X, C_in
        y = self.mlp(_x)  # X, C_out
        y = y.reshape(*input_shape[:-1], self.c_out)
        return y

###############################################################################################

class BaseProjectionLayer(torch.nn.Module):
    @property
    def proj_dims(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class IdentityProjection(BaseProjectionLayer):
    def __init__(self, input_dims):
        super().__init__()
        self._input_dims = input_dims

    @property
    def proj_dims(self):
        return self._input_dims

    def forward(self, x):
        return x


class ProjectionLayer(BaseProjectionLayer):
    def __init__(self, input_dims, proj_dims):
        super().__init__()
        self._proj_dims = proj_dims

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(input_dims, 2 * proj_dims), torch.nn.ReLU(), torch.nn.Linear(2 * proj_dims, proj_dims)
        )

    @property
    def proj_dims(self):
        return self._proj_dims

    def forward(self, x):
        return self.proj(x)


class FixedPositionalEncoding(ProjectionLayer):
    def __init__(self, input_dims, frequency, proj_dims):
        super().__init__(input_dims, proj_dims)
        ll = frequency
        sigma = np.pi * torch.pow(2, torch.linspace(0, ll - 1, ll)).view(1, -1)
        sigma = sigma[None, None, None]
        self.register_buffer("sigma", sigma)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(input_dims + input_dims * ll * 2, proj_dims), torch.nn.LeakyReLU()
        )

    @property
    def proj_dims(self):
        return self._proj_dims * 3

    def forward(self, x):
        encoded = torch.cat(
            [
                torch.sin(x[:, :, :, :, None] * self.sigma),
                torch.cos(x[:, :, :, :, None] * self.sigma),
            ],
            dim=-1,
        ).view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = torch.cat([x, encoded], dim=-1)
        return self.proj(x)


class GaussianRandomFourierFeatures(ProjectionLayer):
    def __init__(self, input_dims, proj_dims, gamma=1.0):
        super().__init__(input_dims, proj_dims)
        self._two_pi = 2 * np.pi
        self._gamma = gamma
        ll = proj_dims // 2
        self.register_buffer("B", torch.randn(3, ll))

    def forward(self, x):
        xB = x.matmul(self.B * self._two_pi * self._gamma)
        return torch.cat([torch.cos(xB), torch.sin(xB)], dim=-1)


class GaborLayer(torch.nn.Module):
    def __init__(self, input_dims, proj_dims, alpha=1., beta=1.0, weight_scale=128):
        super().__init__()
        self.linear = torch.nn.Linear(input_dims, proj_dims)
        self.mu = torch.nn.Parameter(2 * torch.rand(proj_dims, input_dims) - 1)
        self.gamma = torch.nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((proj_dims,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        self.linear2 = torch.nn.Linear(input_dims, proj_dims)
        self.linear2.weight.data.uniform_(
            -np.sqrt(weight_scale / proj_dims),
            np.sqrt(weight_scale / proj_dims)
        )

    def forward(self, x):
        D = (
                (x ** 2).sum(-1)[..., None]
                + (self.mu ** 2).sum(-1)[None, :]
                - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :]) * self.linear2(x)


def get_projection_layer(**kwargs):
    type = kwargs["type"]

    if type == "identity":
        return IdentityProjection(3)
    elif type == "simple":
        return ProjectionLayer(2, kwargs.get("proj_dims", 128))
    elif type == "fixed_positional_encoding":
        return FixedPositionalEncoding(2, kwargs.get("pe_freq", 4), kwargs.get("proj_dims", 128))
    elif type == "gaussianrff":
        return GaussianRandomFourierFeatures(3, kwargs.get("proj_dims", 10), kwargs.get("gamma", 1.0))
    elif type == 'gabor':
        return GaborLayer(3, kwargs.get("proj_dims", 128))
