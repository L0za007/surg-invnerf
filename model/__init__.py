from .mult_filt import GaborNet
from .Hexa_RealNVP import FeatTimeLine, ST_Plane, HexaPlane
from config.config import get_saving_path
from pathlib import Path
import torch

class LeakyReLu(torch.nn.Module):
    def __init__(self, alpha=0.01):
        super(LeakyReLu, self).__init__()
        self.register_buffer('alpha', torch.tensor(alpha))

    def forward(self, x):
        coef = torch.ones_like(x)
        coef[x<0] = self.alpha
        return coef*x

def get_model(config, device, load = False):
    models = {}
    if config.invertible_model == 'InvertNeRF':
        print("Selected model: InvertNeRF")
        from .simple_RealNVP import NVPSimplified
        models['Feature_mlp'] = GaborNet(in_size=1,hidden_size=256,n_layers=2,alpha=4.5,out_size=128)
        models['Deform_mlp'] = NVPSimplified(n_layers=config.n_layers, feature_dims=128, hidden_size=[256, 256, 256],
                                    proj_dims=128, code_proj_hidden_size=[],proj_type='fixed_positional_encoding',
                                    pe_freq=config.pe_freq, normalization=False, affine=config.use_affine,
                                    activation=torch.nn.ReLU,aabb=config.aabb)
    elif config.invertible_model == 'InvertNeRFv2':
        print("Selected model: InvertNeRFv2")
        from .simple_RealNVPv2 import NVPSimplified
        models['Feature_mlp'] = GaborNet(in_size=1,hidden_size=256,n_layers=2,alpha=4.5,out_size=128)
        models['Deform_mlp'] = NVPSimplified(n_layers=config.n_layers,  feature_dims=128, hidden_size=256, hidden_layers=3,
                                    proj_dims=128, proj_type='fixed_positional_encoding', pe_freq=config.pe_freq,  inv_f=config.plane_inv_f)

    elif config.invertible_model == 'fastInvertNeRF':
        print("Selected model: fastInvertNeRF")
        from .fastRealNVP import NVPSimplified
        models['Feature_mlp'] = GaborNet(in_size=1,hidden_size=256,n_layers=2,alpha=4.5,out_size=128)
        models['Deform_mlp'] = NVPSimplified(n_layers=config.n_layers, feature_dims=128,hidden_size=256,hidden_layers=3,
                                    proj_dims=128,proj_type='fixed_positional_encoding',pe_freq=config.pe_freq,)
    
    elif config.invertible_model == 'HexaInvertNeRF':
        print("Selected model: HexaInvertNeRF")
        models['Feature_mlp'] = None
        models['Deform_mlp'] = ST_Plane(n_component=[32,32,32], grid_size=[16,16,16],time_grid=12+config.num_imgs//20, n_res=4,
                                       aabb=config.aabb, init_scale=1e-4,init_shift=0, n_layers=config.n_layers,
                                       inv_f=config.plane_inv_f)
                                       #inv_f='NVPnonlin'
    elif config.invertible_model == 'NVPnonlin':
        from .nvp_nonlin import NVPnonlin
        models['Feature_mlp'] = None
        print(f"Selected model: NVPnonlin - {config.num_imgs} images")
        models['Deform_mlp'] = NVPnonlin(n_layers=config.n_layers,n_frames=config.num_imgs,feature_dim=32, t_dim = 16,
                                        multires=True, base_res=12, net_layer=3, device=device)
    else: 
        raise NotImplementedError(f'Model type {config.invertible_model} not implemented')
    
    if models['Feature_mlp'] is not None and config.t_encoder == 'FeatTimeLine':
        print("Using FeatTimeLine for feature encoding")
        models['Feature_mlp'] = FeatTimeLine(nt_component=128//4,
                nt_grid_size=12+config.num_imgs//8, n_res=4,
                init_scale=0.1, init_shift=0.0, max_t=config.num_imgs)
    
    # A clip shorter than the temporal grid leaves no room for the grid to grow, so the
    # resolution drops to one node per frame. Never binds above 30 frames.
    nt_grid = min(12 + config.num_imgs // 8, config.num_imgs)

    if config.color_model == 'GaborNet':
        models['Color_mlp'] = GaborNet(in_size=config.color_input, hidden_size=256, n_layers=4,alpha=3,out_size=4)
    elif config.color_model == 'HexaPlane':
        models['Color_mlp'] = HexaPlane(n_component=[16,16,16], grid_size=[32,32,32],  nt_grid_size=[nt_grid]*3,
                                          n_res=config.density_n_levels, out_size=3,
                                          init_scale=0.1, init_shift=1e-4, add_t=config.add_time, max_t=max(config.num_imgs, nt_grid))
    elif config.color_model == None:
        models['Color_mlp'] = None
    else: 
        raise NotImplementedError(f'Color model type {config.color_model} not implemented')
    print(f"Color mlp: {config.color_model}")
    
    if config.density_model == 'HexaPlane':
        models['Density_mlp'] = HexaPlane(n_component=[16,16,16], grid_size=[32,32,32],  nt_grid_size=[nt_grid]*3,
                                          n_res=config.density_n_levels, 
                                          init_scale=0.1, init_shift=1e-4, add_t=config.add_time, max_t=max(int(config.num_imgs/2), nt_grid))
    elif config.density_model == None:
        models['Density_mlp'] = None
    else: 
        raise NotImplementedError(f'Density model type {config.density_model} not implemented')
    print(f"Density mlp: {config.density_model}")

    if config.estimator == 'density_est':
        from .ngp import NGPDensityField
        models['Rho_mlp'] = torch.nn.Sequential(NGPDensityField(aabb=config.aabb, num_dim = 4, unbounded=False, n_levels=16, log2_hashmap_size=19, max_resolution=2048))

    if load: 
        if len(config.ckpt_path)>0:
            # Load check point from specified directory
            assert Path(config.ckpt_path).exists(), f'Checkpoint not found: {config.ckpt_path}'
            print(f'Loading checkpoint: {config.ckpt_path}')
            checkpoint = torch.load(config.ckpt_path, map_location=device, weights_only=False)
        else:
            # list the checkpoints in the output directory
            saving_path = get_saving_path(config)
            checkpoints = list(saving_path.glob('model_*.pth'))
            checkpoints.sort()
            if len(checkpoints) > 0:
                checkpoint = checkpoints[-1]
                config.ckpt_path = str(checkpoint)
                print(f'Loading checkpoint: {checkpoint}')
                checkpoint = torch.load(checkpoint, map_location=device)
            else:
                raise FileNotFoundError(str(saving_path)+'/model_*.pth')
        for k, v in models.items():
                if k in checkpoint.keys():
                    v.load_state_dict(checkpoint[k])
                elif v is not None:
                    raise ValueError(f"Invalid network found - {v}")
    return models