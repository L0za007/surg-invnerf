import torch
import numpy as np

class AffineNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, skip_layers=[4], num_layers=8,
                 act='relu', use_pe=False, pe_freq=10, pe_dims=None):
        super(AffineNet, self).__init__()
        """ AffineNet is a fully connected neural network with skip connections and positional encoding.
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
            skip_layers (list): list of layers where skip connections are added
            num_layers (int): number of layers
            act (str): activation function
            use_pe (bool): use positional encoding
            pe_freq (int): frequency of the positional encoding
            pe_dims (list): list of dimensions (coordinates) to apply the positional encoding
        """
        self.pe_dims = pe_dims
        self.skip_layers = skip_layers
        self.num_layers = num_layers
        self.use_pe = use_pe

        if use_pe:
            if pe_dims == None:
                encoding_dimensions = 2 * input_dim * pe_freq + input_dim
            else:
                encoding_dimensions = 2 * len(pe_dims) * pe_freq + input_dim
            self.pos_encod = positionalEncoding_vec(pe_freq)
        else:
            encoding_dimensions = input_dim

        self.hidden = torch.nn.ModuleList()
        for i in range(num_layers):
            # Set input dimensions for layer i
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim
            # Add activation function to all layers except the last one
            if i == num_layers - 1:
                # last layer
                self.hidden.append(torch.nn.Linear(input_dims, output_dim, bias=True))
            else:
                act_ = get_activation_fn(act)
                layer_ = torch.nn.Sequential(torch.nn.Linear(input_dims, hidden_dim, bias=True), act_)
                self.hidden.append(layer_)

    def forward(self, x):
        """ Forward pass of the AffineNet.
        Args:
            x (torch.Tensor): coordinate(s) and time step of the input point. 
                Shape (batch_size, n_pts, n_samples, n_feat)
        Returns:
            torch.Tensor: Parameters of an affine homographic transformation for each point. 
                Shape (batch_size, n_pts, n_samples, 5)
        """
        if self.use_pe:
            # Select the coordinates to apply the positional encoding
            coord = x[..., self.pe_dims] if self.pe_dims is not None else x
            pos = self.pos_encod(coord)
            x = torch.cat([pos, x], dim=-1) # Concatenate the positional encoding with the input

        encoded_input = x
        for i, layer in enumerate(self.hidden):
            if i in self.skip_layers:
                x = torch.cat((x, encoded_input), -1)
            x = layer(x)

        return x

    def get_ATM_from_params(self, theta, inverse=False):
        """ Expands the 5 parameters into 3x3 Affine Homographic Transformation Matrix (ATM)
        Args:
            theta (torch.Tensor): parameters of the affine transformation (theta, scale1, tx, scale2, ty)
                Shape (batch_size, n_pts, n_samples, 5)
            inverse (bool): if True, returns the inverse of the ATM
        Returns:
            | a, b, tx |
            | c, d, ty |
            | 0, 0, 1  |
            torch.Tensor: ATM. Shape (batch_size, n_pts, n_samples, 3, 3)
        """
        angle = theta[..., 0:1]
        scale1 = torch.exp(theta[..., 1:2])
        scale2 = torch.exp(theta[..., 3:4])
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        a = cos * scale1  # NOTE: Here I did a major change in the order of the element. Currently: in the original mode
        b = -sin * scale1
        #b = -sin * scale2
        c = sin * scale2
        d = cos * scale2
        #d = cos * scale1
        tx = theta[..., 2:3]
        ty = theta[..., 4:5]
        zeros = torch.zeros_like(a)
        ones = torch.ones_like(a)
        if inverse:
            return self.invert_ATM(a, b, c, d, tx, ty, zeros, ones)
        else:
            return torch.cat([a, b, tx, c, d, ty, zeros, zeros, ones], dim=-1).reshape(*theta.shape[:-1], 3, 3)

    def invert_ATM(self, a, b, c, d, tx, ty, zeros, ones):
        """ Inverts the Affine Homographic Transformation Matrix (ATM)
        Args:
            | a, b, tx |
            | c, d, ty |
            | 0, 0, 1  |
            All elements have shape (batch_size, n_pts, n_samples)
        Returns:
            torch.Tensor: Inverted ATM. Shape (batch_size, n_pts, n_samples, 3, 3)
        """
        determinant = a * d - b * c

        inverse_determinant = 1.0 / determinant

        inverted_a = d * inverse_determinant
        inverted_b = -b * inverse_determinant
        inverted_c = -c * inverse_determinant
        inverted_d = a * inverse_determinant
        inverted_tx = (b * ty - d * tx) * inverse_determinant
        inverted_ty = (c * tx - a * ty) * inverse_determinant

        return torch.cat([inverted_a, inverted_b, inverted_tx,
                          inverted_c, inverted_d, inverted_ty,
                          zeros, zeros, ones], dim=-1).reshape(*a.shape[:-1], 3, 3)

def apply_TM_xy1(mat, xy1):
    """ Apply transformation matrix to xy1 coordinates
    Args:
        mat (torch.Tensor): transformation matrix. Shape (batch_size, 3, 3)
        xy1 (torch.Tensor): xy1 coordinates. Shape (batch_size, n_pts, n_samples, 3)
    Returns:
        torch.Tensor: warped coordinates. Shape  (batch_size, n_pts, n_samples, 2)
    """
    out_h = torch.matmul(mat, xy1[..., None])
    return out_h[..., :2, 0] / (out_h[..., 2:, 0] + 1e-8)


def apply_TM(mat, uv):
    """ Apply transformation matrix to uv coordinates
    Args:
        mat (torch.Tensor): transformation matrix. Shape (batch_size, n_pts, n_samples 3, 3)
        uv (torch.Tensor): uv coordinates. Shape (batch_size, n_pts, n_samples, 2)
    Returns:
        torch.Tensor: warped coordinates. Shape  (batch_size, n_pts, n_samples, 2)
    """
    uv_h = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)  # (..., 3)
    return apply_TM_xy1(mat, uv_h)

class positionalEncoding_vec(torch.nn.Module):
    def __init__(self, pe_freq):
        super(positionalEncoding_vec, self).__init__()
        self.pe_freq = pe_freq
        b = torch.tensor([(2 ** j) * np.pi for j in range(pe_freq)], requires_grad=False)
        self.register_buffer('b', b)

    def forward(self, in_tensor):
        o_shape = in_tensor.shape            # original shape (batch_s, n_pts, n_samples, n_features)
        flat_size = torch.prod(torch.tensor(o_shape[:-1]))   # all_points: batch_s*n_pts*n_samples
        in_tensor_flatten = in_tensor.reshape(flat_size, -1) #(all_points, n_features)
        proj = torch.einsum('ij, k -> ijk', in_tensor_flatten, self.b)        # shape (all_points,  n_features, freqNum)
        mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  # shape (all_points,2*n_features, freqNum)
        output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1) # shape (all_points, 2*n_features*freqNum)
        output = output.reshape(o_shape[:-1] + (-1,))# shape (batch_s, n_pts, n_samples, 2*n_features*freqNum)
        return output
    
def get_activation_fn(activation):
    """Return the activation function from string."""
    if activation == "relu":
        return torch.nn.ReLU(True)
    elif activation == "elu":
        return torch.nn.ELU(True)
    elif activation == "leakyrelu":
        return torch.nn.LeakyReLU(True)
    elif activation == "gaussian":
        return GaussianActivation(sigma=0.5, trainable=False)
    else:
        raise RuntimeError(f"activation should be relu/gelu/tanh/sigmoid, not {activation}")
    
class GaussianActivation(torch.nn.Module):
    def __init__(self, sigma=1., trainable=True):
        """Gaussian activation function with mean 0 and standard deviation sigma."""
        super().__init__()
        self.register_parameter('sigma', torch.nn.Parameter(sigma*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.sigma**2))
    