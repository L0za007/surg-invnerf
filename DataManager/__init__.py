from torch.utils.data import DataLoader
from torch.nn import functional as F
from .raft import *
import pytorch_lightning as pl
import torch.multiprocessing as mp
from pathlib import Path

def gen_grid(h=None, w=None, normalize=False, homogeneous=False) -> Tensor:
        """ Generate a grid with the position of each pixel
        Args:
            h (int): Height of the grid
            w (int): Width of the grid
            normalize (bool): Normalize the grid between -1 and 1
            homogeneous (bool): Add a homogeneous coordinate to the grid
        Returns:
            grid (Tensor): Grid with the position of each pixel"""
        if normalize:
            lin_y = torch.linspace(-1., 1., steps=h)
            lin_x = torch.linspace(-1., 1., steps=w)
        else:
            lin_y = torch.arange(0, h)
            lin_x = torch.arange(0, w)
        grid_y, grid_x = torch.meshgrid((lin_y, lin_x), indexing='ij') # Addressing torch warning regarding indexing
        grid = torch.stack((grid_x, grid_y), -1)
        if homogeneous:
            grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
        return grid  # [h, w, 2 or 3]

def init_error_map(config, scale=1) -> Tensor:
    img_dir  = Path(config.root_dir) / Path(config.data_dir) / 'color'
    img_names = sorted([f.stem for f in img_dir.glob('*.png')])
    num_imgs = min(config.max_num_imgs, len(img_names))
    h, w, _ = imageio.imread(str(img_dir / (img_names[0]+'.png'))).shape
    size = (int(h*scale), int(w*scale))
    em_step = config.em_step
    pairs_win = min(config.pairs_win, num_imgs - 1)
    map_grid = gen_grid(*size)[::em_step, ::em_step, ...]
    error_map = torch.zeros(num_imgs, pairs_win*2, *map_grid.shape[:2]) + 1e-6
    return error_map

def get_dataset(config, out_path, loaders=True, eval=False):
    # Load the dataset
    error_map = init_error_map(config).share_memory_()
    train_dataset = RAFTDataset(config, out_path, error_map, eval=eval)
    images = train_dataset.left_imgs, train_dataset.right_imgs
    eval_dataset = RAFTDatasetEval(config, out_path, images, eval=eval)

    if loaders: # Create Loaders
        train_loader = DataLoader(train_dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    shuffle=False,

                            )
        eval_loader = DataLoader(eval_dataset,
                    batch_size=config.val_batch_size,
                    num_workers=config.num_workers,
                    shuffle=False
                            )
        return train_dataset, train_loader, eval_dataset, eval_loader
    
    return train_dataset, eval_dataset

class RAFTDataModule(pl.LightningDataModule):
    def __init__(self, config, out_path, train_dataset, eval_dataset):
        super().__init__()
        self.config = config
        self.out_path = out_path
        self.train_dataset = train_dataset 
        self.eval_dataset = eval_dataset
        self.cache_dir = Path(out_path) / 'cache'
        self.scale = config.scale_factor

    def train_dataloader(self):
        n_scale_flags = len(list(self.cache_dir.glob('upscale*.txt')))
        self.train_dataset.num_pts = max(self.config.num_pts // (n_scale_flags+1), 128)
        print(f"-> Train loader. batch size: {self.config.batch_size}, num_pts: {self.train_dataset.num_pts}")
        return DataLoader(self.train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=False
                        )

    def val_dataloader(self):
        # Avoid resetting the dataloader; just update the dataset's internal state
        n_scale_flags = len(list(self.cache_dir.glob('upscale*.txt')))
        scale = min(self.scale * (2**n_scale_flags), self.config.max_scale)
        print("Val loader: ", f"Batch size: {self.config.val_batch_size}, Scale {scale}")
        return DataLoader(self.eval_dataset,
                          batch_size=self.config.val_batch_size,
                          num_workers=self.config.num_workers,
                          shuffle=False)