"""
Simplyfied version of the Real-valued Non-Volume Preserving (RealNVP) model. 
Similar version is used in Omnimotion model.
"""
import numpy as np
import torch
# try:
#     import tinycudann as tcnn
# except ImportError as e:
#     print(
#         f"Error: {e}! "
#         "Please install tinycudann by: "
#         "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
#     )
#     exit()

class NVPSimplified(torch.nn.Module):
    def __init__(self, n_layers, feature_dims, hidden_layers, hidden_size, proj_dims,
        proj_type="simple", pe_freq=4):
        """ Simple RealNVP model.
        Args:
            n_layers (int): Number of coupling layers.
            feature_dims (int):                  Size of the input features.
            proj_dims (int):                     Size of the projection layer.
            proj_type (str):                     Type of projection layer. 
                Options['identity', 'simple', 'fixed_positional_encoding', 'gaussianrff', 'gabor'].
            pe_freq (int):                       Frequency of the positional encoding.
        """
        super().__init__()

        # make layers
        input_dims = 3 # x, y, z
        self.layers = torch.nn.ModuleList()
        self.code_projectors = torch.nn.ModuleList()
        self.layer_idx = list(range(n_layers))
        
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
            # Projection Net for the temporal features in each layer
            # network_config={"otype": "FullyFusedMLP", "activation": "LeakyReLU", "output_activation": "None", "n_neurons": feature_dims, "n_hidden_layers": 1}
            # self.code_projectors.append(MLP(feature_dims, feature_dims, net_config=network_config))
            self.code_projectors.append(MLP( feature_dims, feature_dims, [feature_dims], bn=None, act=torch.nn.ReLU))

            # Building the coupling block of the Real NVP model
            ## proj is the projection layer that uses an encoder to project the input to a higher dimension (example positional encoding)
            proj = get_projection_layer(proj_dims=proj_dims, type=proj_type, pe_freq=pe_freq)
            # network_config={"otype": "CutlassMLP","activation": "LeakyReLU","output_activation": "None","n_neurons": hidden_size[0],"n_hidden_layers": hidden_layers,}
            # map_st = MLP(feature_dims + proj_dims, 2, net_config=network_config)
            map_st = MLP(proj_dims + feature_dims, 2, hidden_layers*[hidden_size], bn=None, act=torch.nn.ReLU)
            mask = masks[i][None, None, None]
            
            self.layers.append(CouplingLayer(map_st, proj, mask))

    def _expand_features(self, F, x):
        """ Expands features to match the input dimensions of x
        :param F: (B, nF)
        :param x: (B, N, K, C)
        :returns: (B, N, K, nF)"""
        _, N, K, _ = x.shape
        return F[:, None, None, :].expand(-1, N, K, -1)

    def forward(self, t, feat, x):
        y = x

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

        return x

class CouplingLayer(torch.nn.Module):
    def __init__(self, map_st, projection, mask):
        super().__init__()
        self.map_st = map_st
        self.projection = projection
        self.register_buffer("mask", mask)

    def forward(self, F, y):
        y1 = y * self.mask
        F_y1 = torch.cat([F, self.projection(y[..., self.mask.squeeze().bool()])], dim=-1)
        st = self.map_st(F_y1)
        s, t = torch.split(st, split_size_or_sections=1, dim=-1)
        s = torch.tanh(s/4) * 4 
        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj

    def inverse(self, F, x):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x[..., self.mask.squeeze().bool()])], dim=-1)
        st = self.map_st(F_x1)
        s, t = torch.split(st, split_size_or_sections=1, dim=-1)
        s = torch.tanh(s/4) * 4
        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)

        return y, ldj

# class MLP(torch.nn.Module):
#     def __init__(self, c_in, c_out, net_config=None):
#         """ Base MLP for non-flat input"""
#         print("using tcnn model")
#         super().__init__()
#         mlp = tcnn.Network(n_input_dims=c_in, n_output_dims=c_out, network_config=net_config)
#         for param in mlp.parameters():
#             torch.nn.init.uniform_(param, a=-6e-2, b=6e-2)
#         self.mlp = mlp
#         self.c_out = c_out

#     def forward(self, x):
#         # x: B,...,C_in
#         input_shape = x.shape
#         C = input_shape[-1]
#         _x = x.reshape(-1, C)  # X, C_in
#         y = self.mlp(_x)  # X, C_out
#         y = y.reshape(*input_shape[:-1], self.c_out)
#         print("MLP output sample: ", y[0,0])
#         return y

class MLP(torch.nn.Module):
    def __init__(self, c_in, c_out, c_hiddens, act=torch.nn.LeakyReLU, bn=None):
        """ Base MLP for projecting temporal features."""
        super().__init__()
        layers = []
        d_in = c_in
        for d_out in c_hiddens:
            layers.append(torch.nn.Linear(d_in, d_out))
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
        sigma = sigma[None, None, None] # (1, 1, 1, 1, ll)
        self.register_buffer("sigma", sigma)
        # network_config={"otype": "FullyFusedMLP", "activation": "LeakyReLU", "output_activation": "LeakyReLU", "n_neurons": proj_dims//2, "n_hidden_layers": 1}
        # self.proj = MLP(input_dims + input_dims * ll * 2, proj_dims, net_config=network_config)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(input_dims + input_dims * ll * 2, proj_dims), torch.nn.LeakyReLU()
        )

    @property
    def proj_dims(self):
        return self._proj_dims * 3

    def forward(self, x): 
        # x: B, N_p, N_s, 2
        encoded = torch.cat(
            [
                torch.sin(x[:, :, :, :, None] * self.sigma),
                torch.cos(x[:, :, :, :, None] * self.sigma),
            ],
            dim=-1,
        ).view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = torch.cat([x, encoded], dim=-1)
        x = self.proj(x)
        return x


class PositionalEncoding(torch.nn.Module):
        def __init__(self, input_dims, frequency, proj_dims):
            super().__init__()
            # Pre-compute the scaling factor for the frequencies
            sigma = torch.exp(torch.arange(0, frequency, 2).float() * 
                torch.log(torch.tensor(100.0)) / frequency)
            sigma = sigma[None, None, None]
            self.register_buffer("sigma", sigma)
            self.freq = frequency
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(input_dims + 2 * frequency, proj_dims), torch.nn.LeakyReLU()
            )

        def forward(self, xy):
            x_positions,y_positions = xy.split(1, dim=-1)
            # Calculate the sinusoidal encoding for x and y using continuous positions
            x_pe = torch.zeros(*x_positions.shape[:-1], self.freq, device=x_positions.device)
            y_pe = torch.zeros(*y_positions.shape[:-1], self.freq, device=y_positions.device)

            # Apply sine and cosine functions based on the continuous position
            x_pe[..., 0::2] = torch.sin(x_positions * self.sigma)
            x_pe[..., 1::2] = torch.cos(x_positions * self.sigma)
            y_pe[..., 0::2] = torch.sin(y_positions * self.sigma)
            y_pe[..., 1::2] = torch.cos(y_positions * self.sigma)

            # Concatenate x and y encodings along the last dimension
            pe = torch.cat((x_positions, x_pe, y_positions, y_pe), dim=-1)  # Shape: [batch_size, seq_len, d_model]
            
            return self.proj(pe)

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
    elif type == 'positional_encoding':
            return PositionalEncoding(2, kwargs.get("pe_freq", 4), kwargs.get("proj_dims", 128))
