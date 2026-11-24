"""
Simplyfied version of the Real-valued Non-Volume Preserving (RealNVP) model. 
Similar version is used in Omnimotion model.
"""
import torch
from torch.nn import functional as F

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

        # try:
        dYdqx = (yr - yl) / x_range
        dYdyr = (qx[..., 0] - xl) / x_range
        dYdyl = (xr - qx[..., 0]) / x_range
        dYdxr = -dYdyr * (yr - yl) / x_range
        dYdxl = -dYdyl * (yr - yl) / x_range
        # except:
            # pdb.set_trace()
            # i = torch.where(torch.sum(in_range, dim=-1) == 0)
            # i = i[0][0]
            # print("i", i)
            # print("qx", qx[i])
            # print("xl", all_xl[i])
            # print("xr", all_xr[i])

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


class ST_Plane(torch.nn.Module):
    def __init__(self, n_component, grid_size, n_res=3, n_layers=1,
                 nt_component=None, nt_grid_size=None, max_t=200,
                 init_scale=0.1, init_shift=0.0, inv_f='NVP'):
        """ Simple RealNVP model.
        Args:
            n_res (int): Number of plane resolutions.
            feature_dims (int):                  Size of the input features.
            proj_dims (int):                     Size of the projection layer.
            code_proj_hidden_size (list of int): Size of the hidden layers for the code projector.
            proj_type (str):                     Type of projection layer. 
                Options['identity', 'simple', 'fixed_positional_encoding', 'gaussianrff', 'gabor'].
            pe_freq (int):                       Frequency of the positional encoding.
            normalization (bool):                Whether to use batch normalization.
            activation (torch.nn.Module):        Activation function.
        """
        super().__init__()

        # Planes
        self.matMode = [[0, 1], [0, 2], [1, 2]] # xy, xz, yz
        self.vecMode = [2, 1, 0]                #  z,  y,  x
        # Planes params
        self.grid_size   = grid_size   # [32, 32, 32]
        self.n_component = n_component # [3, 3, 3]
        self.init_scale  = init_scale
        self.init_shift  = init_shift
        self.align_corners = True
        self.fusion_op   = "concat"
        self.max_g_size = 450
        self.max_t = max_t 
        nt_component = nt_component if nt_component is not None else n_component
        nt_grid_size = nt_grid_size if nt_grid_size is not None else grid_size
        
        # Coupling layers
        self.n_layers = n_layers
        input_dims = 3 # x, y, z
        self.n_levels = n_res
        self.inv_f = inv_f
        mapper_out, mapper_act = 2, None
        if self.inv_f == 'NVPnonlin':
            mapper_out = 10
            mapper_act = torch.nn.Sigmoid
        print(f"inv_f: {self.inv_f}, mapper_out: {mapper_out}, mapper_act: {mapper_act}")

        # Temporal lines
        time_lines = []
        for i in range(input_dims):
            time_line = self.init_one_time_line(i, nt_component, None, nt_grid_size)
            time_lines.append(time_line)
        self.time_line = torch.nn.ParameterList(time_lines)
        # Spatial planes
        self.layer_idx = list(range(input_dims*n_layers))
        planes = []
        self.st_mappers = torch.nn.ModuleList()
        for i in self.layer_idx:
            plane_idx = i % input_dims
            plane = self.init_one_STplane(plane_idx, n_component, grid_size, None)
            planes.append(plane)
            # Building block of the Real NVP model
            feat_dim = n_component[plane_idx] * n_res + nt_component[plane_idx] * n_res
            hidden_size = [feat_dim//2, feat_dim//4]
            map_st = MLP(feat_dim, mapper_out, out_act=mapper_act, c_hiddens=hidden_size)
            self.st_mappers.append(map_st)
        self.plane = torch.nn.ParameterList(planes)  # [n_layer, n_res, n_component, grid_size, grid_size]    
    
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

        return ST_Plane.get_xy_compute(dxl1, dxl2, dxr1, dxr2, dyl1, dyl2, dyr1, dyr2, kl, kr)

    def init_one_STplane(self, plane_idx, n_component, gridSize, time_grid):
        plane_coef = []
        mat_id_0, mat_id_1 = self.matMode[plane_idx]
        gs0,gs1 = gridSize[mat_id_0], gridSize[mat_id_1]
        for grid_scale in range(self.n_levels):
            gs0+= ((self.max_g_size-gs0)**(1/self.n_levels))** (grid_scale+1)
            gs1+= ((self.max_g_size-gs1)**(1/self.n_levels))** (grid_scale+1)
            # initialization of the planes
            _plane = torch.randn((1, n_component[plane_idx], int(gs0), int(gs1)))
            plane_coef.append(torch.nn.Parameter(self.init_scale * _plane + self.init_shift))

        return torch.nn.ParameterList(plane_coef)
    
    def init_one_time_line(self, plane_idx, n_component, gridSize, time_grid):
        line_time_coef = []
        max_g_size = max(self.max_t, time_grid[plane_idx]+1)
        tgs = time_grid[plane_idx]
        for grid_scale in range(self.n_levels):
            tgs+= ((max_g_size-tgs)**(1/self.n_levels))** (grid_scale+1)
            # initialization of the temp lines
            _time_line = torch.randn((1, n_component[plane_idx], 1, int(tgs)))
            line_time_coef.append(torch.nn.Parameter(self.init_scale * _time_line + self.init_shift))

        return torch.nn.ParameterList(line_time_coef)
    
    def sample_hexplane(self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor, 
                        plane_idx: int, s_plane:torch.Tensor, t_line:torch.Tensor) -> torch.Tensor:
        """
        Compuate the app features of sampled points from appearance HexPlane.

        Args:
            xyz_sampled: (B, NP, NS, 3) sampled points' xyz coordinates.
            frame_time : (B, 1) sampled points' frame time.
            plane_idx  : 

        Returns:
            app_feature: (N, self.app_dim) density of sampled points.
        """
        # Prepare coordinates for grid sampling.
        B, NP, NS, _ = xyz_sampled.shape
        coord_plane = self.matMode[plane_idx] # Coordinate plane. [0, 1], [0, 2], [1, 2] => [[x, y], [x,z], [y,z]].
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        plane_coord = xyz_sampled[..., coord_plane].reshape(-1,2) # [B*NP*NS, 2]
        
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t], [t], [t]].
        line_time_coord = torch.ones_like(xyz_sampled[..., [0]])*0.5
        frame_time = frame_time[...,None,None].expand(line_time_coord.shape)
        line_time_coord = torch.cat([frame_time, line_time_coord], dim=-1).reshape(-1,2) # [B*NP*NS, 2]
        
        
        # Move coordinates from unit cube to the range of the plane [-1,1].
        plane_coord = 2 * plane_coord - 1         # [B, NP*NS, 2]
        line_time_coord = 2 * line_time_coord - 1 # [B, NP*NS, 2]
        # print(f"Point to sample on the spatial plane ({coord_plane})",plane_coord[0,0].cpu().numpy())
        # print(f"Point to sample on the spatial-temporal plane ({st_plane})",line_time_coord[0,0].cpu().numpy())

        # Sample on the HexPlane.
        s_feats,t_feats = [], []
        for i in range(self.n_levels):
            plane = s_plane[i] # [1, F, H, W]
            line_time = t_line[i] # [1, F, 1, T]
            _feat = F.grid_sample(plane, plane_coord[None, :, None], align_corners=self.align_corners) # [1, F, B*NP*NS, 1]
            s_feats.append(_feat.squeeze(0).squeeze(-1).T) # [B*NP*NS, F]
            _t_feat = F.grid_sample(line_time, line_time_coord[None, :, None], align_corners=self.align_corners)  # [1, F, B*NP*NS, 1]
            t_feats.append(_t_feat.squeeze(0).squeeze(-1).T) # [B*NP*NS, F]
        # Multi resolution fussion option is always concat
        s_feats = torch.cat(s_feats, dim=-1) # [B*NP*NS, F]
        t_feats = torch.cat(t_feats, dim=-1) # [B*NP*NS, F]
        # print("time feats shape", t_feats.shape)
        # print("spatial feats shape", s_feats.shape)
        # print("Time feats", t_feats[:,0,0])
        # Fusion One
        if self.fusion_op == "multiply":
            inter = s_feats * t_feats
        elif self.fusion_op == "sum":
            inter = s_feats + t_feats
        elif self.fusion_op == "concat":
            inter = torch.cat([s_feats, t_feats], dim=-1)
        else:
            raise NotImplementedError("no such fusion type")

        # print("inter shape", inter.shape)
        inter = inter.reshape(B, NP, NS, -1) # [B, NP, NS, F]
    
        return inter

    def forward(self, xyz, time):
        for i in self.layer_idx:
            # print(f"Layer {i}, xyz shape: {xyz.shape}, time shape: {time.shape}")
            # print("xyz[0,:8,0,:]:", xyz[0,:8,0,:].T)
            s_plane = self.plane[i]
            p_idx = i % 3
            t_line = self.time_line[p_idx]
            feats = self.sample_hexplane(xyz,time, p_idx, s_plane, t_line) 
            st = self.st_mappers[i](feats) # order of the layers ((sZ,tZ) (sY,tY) (sX,tX))
            mask = torch.zeros_like(xyz)
            mask[..., self.vecMode[p_idx]] = 1
            # print("Modifying xyz with mask", mask[0][:8].T)
            if self.inv_f == 'NVP':
                s,t = torch.split(st, split_size_or_sections=1, dim=-1)
                s = torch.tanh(s) * 4
                # t = torch.tanh(t) * self.max_t[p_idx]/2
                xyz = xyz*(1-mask) + mask*(xyz*torch.exp(s) + t)
            else:
                all_x, all_y = self.get_all_xy(st)
                all_x, all_y = all_x.reshape(-1, 6), all_y.reshape(-1, 6)
                _x = xyz[..., self.vecMode[p_idx]].reshape(-1, 1)
                _y = interp_grad(all_x, all_y, _x).reshape(*xyz.shape[:-1])
                xyz = xyz*(1-mask)
                # print("xyz[0][:8]:", xyz[0][:8].T)
                xyz[..., self.vecMode[p_idx]] += _y
            
            # print("xyz[0,:8,0,:]:", xyz[0,:8,0,:].T)
        return xyz

    def inverse(self, canonical_xyz, time):
        for i in self.layer_idx[::-1]:
            # print(f"Layer {i}, xyz shape: {canonical_xyz.shape}, time shape: {time.shape}")
            # print("canonical_xyz[0,:8,0,:]:", canonical_xyz[0,:8,0,:].T)
            s_plane = self.plane[i]
            p_idx = i % 3
            t_line = self.time_line[p_idx]
            feats = self.sample_hexplane(canonical_xyz,time, p_idx, s_plane, t_line)
            st = self.st_mappers[i](feats) # order of the layers ((sX,tX) (sY,tY) (sZ,tZ))
            mask = torch.zeros_like(canonical_xyz)
            mask[..., self.vecMode[p_idx]] = 1
            # print("modifying canonical_xyz with mask", mask[0][:8].T)
            if self.inv_f == 'NVP':
                s,t = torch.split(st, split_size_or_sections=1, dim=-1)
                s = torch.tanh(s) * 4
                # t = torch.tanh(t) * self.max_t[p_idx]/2
                canonical_xyz = canonical_xyz*(1-mask) + mask*(canonical_xyz - t) * torch.exp(-s)
            else:
                all_x, all_y = self.get_all_xy(st)
                all_x, all_y = all_x.reshape(-1, 6), all_y.reshape(-1, 6)
                _y = canonical_xyz[..., self.vecMode[p_idx]].reshape(-1, 1)
                _x = interp_grad(all_y, all_x, _y).reshape(*canonical_xyz.shape[:-1])
                canonical_xyz = canonical_xyz*(1-mask)
                canonical_xyz[..., self.vecMode[p_idx]] += _x
            # print("xyz[0][:8]:", canonical_xyz[0][:8].T)
        return canonical_xyz


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
        if self.out_act is not None:
            y = self.out_act(y)
        y = y.reshape(*input_shape[:-1], self.c_out)
        return y
    

class FeatTimeLine(torch.nn.Module):
    def __init__(self, nt_component=None, nt_grid_size=None, n_res=3, 
                init_scale=0.1, init_shift=0.0, max_t=200):
        super().__init__()

        self.init_scale    = init_scale
        self.init_shift    = init_shift
        self.align_corners = True
        self.fusion_op     = "concat"
        self.n_levels      = n_res
        self.max_t         = max_t

        # Temporal lines
        t_lines = []
        for grid_scale in range(n_res):
            t_line = self.init_one_time_line(nt_component, nt_grid_size, grid_scale)
            t_lines.append(t_line)
        self.t_line = torch.nn.ParameterList(t_lines)
    
    def init_one_time_line(self, n_component, gridSize, grid_scale):
        max_g_size = max(self.max_t, gridSize+1)
        tgs = gridSize + ((max_g_size-gridSize)**(1/self.n_levels))** (grid_scale+1)
        _time_line = torch.randn((1, n_component, 1, int(tgs)))
        _time_line = torch.nn.Parameter(self.init_scale * _time_line + self.init_shift)
        return _time_line
    
    def sample_time_line(self,time: torch.Tensor) -> torch.Tensor:
        # Prepare coordinates for grid sampling.
        B, _ = time.shape
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t], [t], [t]].
        line_time_coord = torch.ones_like(time)*0.5
        line_time_coord = torch.cat([time, line_time_coord], dim=-1).unsqueeze(-2)  # [B, 1, 2]
        # Move coordinates from unit cube to the range of the plane [-1,1].
        line_time_coord = 2 * line_time_coord - 1
        # print(f"line_time_coord shape: {line_time_coord.shape}, line_time_coord: {line_time_coord[0].cpu().numpy()}")
        # Sample the temporal line.
        t_feats = []
        for i in range(self.n_levels):
            t_feat = F.grid_sample(self.t_line[i], line_time_coord[None], align_corners=self.align_corners)
            t_feats.append(t_feat.squeeze(0).squeeze(-1).T) # [B, n_component]
        t_feats = torch.cat(t_feats, dim=-1)   # [B, n_component * n_levels]
        return t_feats

    def forward(self, time):
        feat = self.sample_time_line(time)
        return feat


class HexaPlane(torch.nn.Module):
    def __init__(self, n_component, grid_size, n_res=3, 
                 nt_component=None, nt_grid_size=None,
                 init_scale=0.1, init_shift=0.0, add_t=True, max_t=200, out_size = 1):
        super().__init__()

        # Planes
        self.matMode = [[0, 1], [0, 2], [1, 2]] # xy, xz, yz
        self.vecMode = [2, 1, 0]                #  z,  y,  x
        # Planes params
        self.grid_size   = grid_size   # [32, 32, 32]
        self.align_corners = True
        self.init_scale  = init_scale
        self.init_shift  = init_shift
        self.fusion_op   = "concat"
        self.n_levels = n_res
        self.max_g_size = 450
        self.max_t = max_t 

        density_planes = []
        for grid_scale in range(n_res):
            density_plane = self.init_one_plane(n_component, grid_size, grid_scale)
            density_planes.append(density_plane)
        self.density_plane = torch.nn.ParameterList(density_planes)
        feat_size = n_res * n_component[0] * 3 if self.fusion_op == "concat" else n_res * n_component[0]

        # Temporal lines
        self.add_t=add_t
        if self.add_t:
            nt_grid_size = nt_grid_size if nt_grid_size is not None else grid_size
            nt_component = nt_component if nt_component is not None else n_component
            spatio_temporal_planes = []
            for grid_scale in range(n_res):
                st_plane = self.init_one_spatial_temporal_plane(nt_component, grid_size, nt_grid_size, grid_scale)
                spatio_temporal_planes.append(st_plane)
            self.st_planes = torch.nn.ParameterList(spatio_temporal_planes)
            feat_size += n_res * nt_component[0] * 3 if self.fusion_op == "concat" else n_res * nt_component[0]
        
        # Mapper
        hidden_size = [feat_size, feat_size//2, feat_size//2, feat_size//4]
        self.density_projector = MLP(feat_size, out_size, out_act=None, c_hiddens=hidden_size)

    def init_one_plane(self, n_component, gridSize, grid_scale):
        plane_coef = []
        grid_scale+=1
        for i in range(len(self.matMode)):
            mat_id_0, mat_id_1 = self.matMode[i]
            gs0,gs1 = gridSize[mat_id_0], gridSize[mat_id_1]
            gs0+= ((self.max_g_size-gs0)**(1/self.n_levels))** grid_scale
            gs1+= ((self.max_g_size-gs1)**(1/self.n_levels))** grid_scale
            # initialization of the planes
            _plane = torch.randn((1, n_component[i], int(gs1),int(gs0)))
            plane_coef.append(torch.nn.Parameter(self.init_scale * _plane + self.init_shift))

        return torch.nn.ParameterList(plane_coef)
    
    def init_one_spatial_temporal_plane(self, n_component, gridSize, tgridSize, grid_scale):
        st_coef= []
        grid_scale+=1
        max_g_size = self.max_t
        for i in range(len(self.vecMode)):
            axes = self.vecMode[i]
            gs,tgs = gridSize[axes], tgridSize[axes]
            gs+= ((self.max_g_size-gs)**(1/self.n_levels))** grid_scale
            tgs+= ((max_g_size-tgs)**(1/self.n_levels))** grid_scale
            st_plane = torch.randn((1, n_component[i], int(gs), int(tgs)))
            st_plane = torch.nn.Parameter(self.init_scale * st_plane + self.init_shift)
            st_coef.append(st_plane)
        
        return torch.nn.ParameterList(st_coef)
    
    def sample_plane(self, xyz_sampled: torch.Tensor, layer_idx: int, plane) -> torch.Tensor:
        """
        Compuate the app features of sampled points from appearance HexPlane.

        Args:
            xyz_sampled: (B, NP, NS, 3) sampled points' xyz coordinates.
            frame_time : (B, 1) sampled points' frame time.
            layer_idx  : (int) index of the layer.

        Returns:
            app_feature: (N, self.app_dim) density of sampled points.
        """
        # Prepare coordinates for grid sampling.
        B, NP, NS, _ = xyz_sampled.shape
        coord_plane = self.matMode[layer_idx] # Coordinate plane. [0, 1], [0, 2], [1, 2] => [[x, y], [x,z], [y,z]].

        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        plane_coord = xyz_sampled[..., coord_plane].reshape(B,NP*NS,-1) # [B, NP*NS, 2]
        
        # Move coordinates from unit cube to the range of the plane [-1,1].
        plane_coord = 2 * plane_coord - 1

        # Sample on the HexPlane.
        s_feats = []
        for i in range(self.n_levels):
            _plane = plane[i][layer_idx]
            s_feats.append(F.grid_sample(_plane, plane_coord[None], align_corners=self.align_corners))
        s_feats = torch.cat(s_feats, dim=0)   # [n_component, B, 1, NP*NS]

        s_feats = s_feats.squeeze().reshape(-1, B, NP, NS) # [n_component, B, NP, NS]
        return s_feats
    
    def sample_st_plane(self, xyz_sampled: torch.Tensor, layer_idx: int, time: torch.Tensor) -> torch.Tensor:

        # Prepare coordinates for grid sampling.
        B, NP, NS, _ = xyz_sampled.shape
        coord_plane = self.vecMode[layer_idx] # Coordinate plane
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t], [t], [t]].
        coord = xyz_sampled[..., [coord_plane]]
        frame_time = time[...,None,None].expand(coord.shape)
        coord = torch.cat([frame_time, coord], dim=-1).reshape(B,NP*NS,-1)
        # Move coordinates from unit cube to the range of the plane [-1,1].
        coord = 2 * coord - 1
        # Sample the temporal line.
        t_feat = []
        for i in range(self.n_levels):
            _st_plane = self.st_planes[i][layer_idx]
            t_feat.append(F.grid_sample(_st_plane, coord[None], align_corners=self.align_corners))
        t_feat = torch.cat(t_feat, dim=0)
        t_feat = t_feat.squeeze().reshape(-1, B, NP, NS)
        return t_feat

    def forward(self, xyz,time):
        density_feat = []
        for i in range(3):
            den_feats = self.sample_plane(xyz, i, self.density_plane)
            density_feat.append(den_feats)
        if self.fusion_op == "concat":
            density_feat = torch.cat(density_feat, dim=0)
        # elif self.fusion_op == "sum":
        #     density_feat = sum(density_feat)
        # elif self.fusion_op == "multiply":
        #     density_feat = torch.prod(density_feat, dim=0)
        else:
            raise NotImplementedError("no such fusion type")
        
        # Sample on the temporal line.
        if self.add_t:
            t_feat=[]
            for i in range(3):
                t_feat.append(self.sample_st_plane(xyz, i, time))
            t_feat=torch.cat(t_feat, dim=0)
            feat = torch.cat([density_feat, t_feat], dim=0)
        else:
            feat = density_feat
        # Reorder the dimensions. [B, NP, NS, n_component]
        feat = feat.permute(1, 2, 3, 0)
        density = self.density_projector(feat)

        return density



