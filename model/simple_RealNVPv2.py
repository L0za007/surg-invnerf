"""
Simplyfied version of the Real-valued Non-Volume Preserving (RealNVP) model. 
Similar version is used in Omnimotion model.
"""
import numpy as np
import torch

class InterpGrad(torch.autograd.Function):

    @staticmethod
    @torch.jit.script
    def forw_compute(x, y, qx):
        """ 
        x: [N, n_samples], should be monotonic (xL3, xL2, xL1, xR1, xR2, xR3)
        y: [N, n_samples]                      (yL3, yL2, yL1, yR1, yR2, yR3)
        qx: [N, 1]
        """
        grad_qx = torch.zeros_like(qx)
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(y)
        out = torch.zeros_like(qx)
        all_xr = x[..., 1:]     #[xL2, xL1, xR1, xR2, xR3]
        all_xl = x[..., :-1]    #[xL3, xL2, xL1, xR1, xR2]
        all_yr = y[..., 1:]     #[yL2, yL1, yR1, yR2, yR3]
        all_yl = y[..., :-1]    #[yL3, yL2, yL1, yR1, yR2]
        qx = torch.clamp(qx, all_xl[..., 0:1]*0.99, all_xr[..., -1:]*0.99) # clap between xL3 and xR3
        in_range = (qx >= all_xl) & (qx < all_xr) # N, 5
        xr = all_xr[in_range] 
        xl = all_xl[in_range] 
        yr = all_yr[in_range]
        yl = all_yl[in_range]
        x_range = xr - xl

        dYdqx = (yr - yl) / x_range
        dYdyr = (qx[..., 0] - xl) / x_range
        dYdyl = (xr - qx[..., 0]) / x_range
        dYdxr = -dYdyr * (yr - yl) / x_range
        dYdxl = -dYdyl * (yr - yl) / x_range

        grad_qx[..., 0] += dYdqx
        grad_x[..., :-1][in_range] += dYdxl
        grad_x[..., 1:][in_range] += dYdxr
        grad_y[..., :-1][in_range] += dYdyl
        grad_y[..., 1:][in_range] += dYdyr
        out[..., 0] = dYdqx * (qx[..., 0] - xl) + yl
        return out, grad_x, grad_y, grad_qx

    @staticmethod
    def forward(ctx, x, y, qx):
        '''
        x: [N, n_samples], should be monotonic
        y: [N, n_samples]
        qx: [N, 1]
        '''
        N, n_samples = x.shape
        out, grad_x, grad_y, grad_qx = InterpGrad.forw_compute(x, y, qx)
        
        ctx.save_for_backward(grad_x, grad_y, grad_qx)

        return out

    @staticmethod
    @torch.jit.script
    def back_compute(grad_out, grad_x, grad_y, grad_qx):
        grad_x = grad_x * grad_out
        grad_y = grad_y * grad_out
        grad_qx = grad_qx * grad_out
        return grad_x, grad_y, grad_qx

    @staticmethod
    def backward(ctx, grad_out):
        grad_x, grad_y, grad_qx = ctx.saved_tensors
        # grad_x = grad_x * grad_out
        # grad_y = grad_y * grad_out
        # grad_qx = grad_qx * grad_out
        grad_x, grad_y, grad_qx = InterpGrad.back_compute(grad_out, grad_x, grad_y, grad_qx)
        return grad_x, grad_y, grad_qx
    
interp_grad = InterpGrad.apply

class NVPSimplified(torch.nn.Module):
    def __init__(self, n_layers, feature_dims, hidden_layers, hidden_size, proj_dims,
        proj_type="simple", pe_freq=4, inv_f='NVP'):
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

        #mapper params
        self.inv_f = inv_f
        map_out, map_act = 2, None
        if self.inv_f == 'NVPnonlin':
            map_out = 10
            map_act = torch.nn.Sigmoid
        print(f"Invertible function: {self.inv_f}, num parameters: {map_out}, out activation: {map_act}")

        for i in self.layer_idx:
            # Projection Net for the temporal features in each layer
            self.code_projectors.append(MLP( feature_dims, feature_dims, [feature_dims], bn=None, act=torch.nn.LeakyReLU))
            # Projection Layer that uses an encoder to project the input to a higher dimensional space (example positional encoding)
            proj = get_projection_layer(proj_dims=proj_dims, type=proj_type, pe_freq=pe_freq)
            # Mapping layer to predict the transformation parameters
            map_st = MLP(proj_dims + feature_dims, map_out, hidden_layers*[hidden_size], bn=None, act=torch.nn.LeakyReLU, out_act=map_act)
            mask = masks[i][None, None, None] 
            
            self.layers.append(CouplingLayer(map_st, proj, mask, inv_f=inv_f))

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
    def __init__(self, map_st, projection, mask, inv_f='NVP'):
        super().__init__()
        self.inv_f = inv_f
        self.map_st = map_st
        self.projection = projection
        self.register_buffer("mask", mask)

    @staticmethod
    @torch.jit.script
    def get_xy_compute(dxl1, dxl2, dxr1, dxr2, dyl1, dyl2, dyr1, dyr2, kl, kr):
        kl = kl * 2.0# + 1e-8
        kr = kr * 2.0# + 1e-8

       
        xL1 = -dxl1
        xL2 = -dxl1 - dxl2
        yL1 = - dyl1
        yL2 = - dyl1 - dyl2

        xR1 = dxr1
        xR2 = dxr1 + dxr2
        yR1 = dyr1
        yR2 = dyr1 + dyr2


        xR3 = xR2 + 1e4
        xL3 = xL2 - 1e4
        yR3 = yR2 + kr*1e4
        yL3 = yL2 - kl*1e4

        all_x = torch.cat([xL3, xL2, xL1, xR1, xR2, xR3], dim=-1)
        all_y = torch.cat([yL3, yL2, yL1, yR1, yR2, yR3], dim=-1)
        return all_x, all_y
    
    def get_all_xy(self, dxdykk):
        '''
        dxdykk: [N, P, 1, 10], dxl1, dxl2, dxr1, dxr2, dy...., kl, kr, all > 0
        '''
        dxl2, dxl1, dxr1, dxr2, dyl2, dyl1, dyr1, dyr2, kl, kr = torch.split(
            dxdykk, split_size_or_sections=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dim=-1)

        return CouplingLayer.get_xy_compute(dxl1, dxl2, dxr1, dxr2, dyl1, dyl2, dyr1, dyr2, kl, kr)

    def forward(self, F, y):
        y1 = y * self.mask
        F_y1 = torch.cat([F, self.projection(y[..., self.mask.squeeze().bool()])], dim=-1)
        st = self.map_st(F_y1)
        if self.inv_f == 'NVP':
            s, t = torch.split(st, split_size_or_sections=1, dim=-1)
            s = torch.tanh(s/8) 
            x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
            ldj = (-s).sum(-1)
        elif self.inv_f == 'NVPnonlin':
            x = y1
            all_x, all_y = self.get_all_xy(st)
            all_x, all_y = all_x.reshape(-1, 6), all_y.reshape(-1, 6)
            query = y[..., (1-self.mask).squeeze().bool()].reshape(-1, 1)
            grad_y = interp_grad(all_x, all_y, query).reshape(*x.shape[:-1], 1)
            x[..., (1-self.mask).squeeze().bool()] = grad_y
            ldj = None

        return x, ldj

    def inverse(self, F, x):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x[..., self.mask.squeeze().bool()])], dim=-1)
        st = self.map_st(F_x1)
        if self.inv_f == 'NVP':
            s, t = torch.split(st, split_size_or_sections=1, dim=-1)
            s = torch.tanh(s/8)
            y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
            ldj = s.sum(-1)
        elif self.inv_f == 'NVPnonlin':
            y = x1
            all_x, all_y = self.get_all_xy(st)
            all_x, all_y = all_x.reshape(-1, 6), all_y.reshape(-1, 6)
            query = x[..., (1-self.mask).squeeze().bool()].reshape(-1, 1)
            grad_x = interp_grad(all_y, all_x, query).reshape(*x.shape[:-1], 1)
            y[..., (1-self.mask).squeeze().bool()] = grad_x
            ldj = None

        return y, ldj

class MLP(torch.nn.Module):
    def __init__(self, c_in, c_out, c_hiddens, act=torch.nn.LeakyReLU, bn=None, out_act=None):
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
        self.out_act = out_act() if out_act is not None else None

    def forward(self, x):
        # x: B,...,C_in
        input_shape = x.shape
        C = input_shape[-1]
        _x = x.reshape(-1, C)  # X, C_in
        y = self.mlp(_x)  # X, C_out
        y = y.reshape(*input_shape[:-1], self.c_out)
        if self.out_act is not None:
            y = self.out_act(y)
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


