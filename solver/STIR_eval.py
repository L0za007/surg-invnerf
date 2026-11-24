# Description: Evaluate tracker on each video os the STIR dataset
import warnings
warnings.filterwarnings("ignore")
import argparse, json
from pathlib import Path
import numpy as np
import sys, re, traceback
from scipy.spatial import KDTree
from PIL import Image, ImageDraw
import imageio

# include local modules
import sys, torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
sys.path.append(str(Path(__file__).parent.parent))
from config.config import get_config, get_saving_path
from DataManager import get_dataset
from DataManager.ray_manager import RayManager
from model import get_model
from engine import BaseTrainer
from vis import get_video_correspondences, get_rgb_and_depth_video

def save_img_w_marks(file_path, marks, out_path, dot_size=1):
    """Save image with marks"""
    img = Image.open(file_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    for pt in marks:
        draw.ellipse([pt[0]-dot_size, pt[1]-dot_size, pt[0]+dot_size, pt[1]+dot_size], fill='red')
    img.save(out_path)

@torch.no_grad()
def run_tracker(marks, out_dir, chain=False):
    """Apply tracker to the marks"""
    # Load the model
    config = get_config(str(out_dir / 'config.yaml'), mk_out_dir=False)
    saving_path = get_saving_path(config)
    dataset, _, _, _ = get_dataset(config, saving_path, eval=True)
    handler = RayManager(config, dataset)
    models = get_model(config, handler.device, load = True)
    framework = BaseTrainer(config, models, 
                                     criterion=None, 
                                     label_manager=handler, 
                                     save_path=str(out_dir)).to(handler.device)
    # Prepare the points
    pts = torch.tensor(marks, device=handler.device, dtype=torch.float32)
    # draw point on starting mask
    try:
        save_img_w_marks(out_dir.parents[1] / 'labels' / 'original_start.png', pts.cpu().numpy(), out_dir / 'mask_start_org.png', dot_size=3)
    except FileNotFoundError:
        print(f"File {'original_start.png'} not found. Skipping saving original start mask.")
    # Scaling the points
    o_h,o_w = dataset.camera_model.orig_h, dataset.camera_model.orig_w
    h,w = dataset.camera_model.h, dataset.camera_model.w
    pts = dataset.camera_model.normalize_coords(pts, o_h, o_w)
    pts = dataset.camera_model.denormalize_coords(pts, h, w)
    # save_img_w_marks(out_dir.parents[1] / 'labels' / 'start.png', pts.cpu().numpy(), out_dir / 'mask_start.png')
    # Video correspondences
    get_video_correspondences(framework, 0, dataset, query_pts=pts, out_dir=out_dir, save_first_last=True, radius=4, save_frames=True)
    get_rgb_and_depth_video(framework, grid_step=1, colorise=True, out_dir=out_dir,save_frames=True, camera_side='right')
    idx1, idx2 = torch.tensor(0, device=handler.device), torch.tensor(len(dataset)-1, device=handler.device)
    # Make inference
    pred_ptsB, occ_mask, VF_mask, pred_3DptsA, pred_3DptsB  = framework.get_corr_pair(idx1, idx2, pts, return3D=True)
    pred_ptsB = pred_ptsB[:,:-1] # Removing the last column which is the occlusion mask
    # save_img_w_marks(out_dir.parents[1] / 'labels' / 'final.png', pred_ptsB.cpu().numpy(), out_dir / 'mask_end.png')
    pred_ptsB = dataset.camera_model.normalize_coords(pred_ptsB, h, w)
    pred_ptsB = dataset.camera_model.denormalize_coords(pred_ptsB, o_h, o_w)
    # draw point on ending mask
    try:
        save_img_w_marks(out_dir.parents[1] / 'labels' / 'original_final.png', pred_ptsB.cpu().numpy(), out_dir / 'mask_original_end.png', dot_size=3)
    except FileNotFoundError:
        print(f"File {'original_final.png'} not found. Skipping saving original end mask.")
    pred_3DptsA = pred_3DptsA * 1000 # Convert to mm
    pred_3DptsB = pred_3DptsB * 1000 # Convert to mm
    
    # Format the output
    return pred_ptsB.cpu().numpy(), pred_3DptsA.cpu().numpy(), pred_3DptsB.cpu().numpy(), config

def extract_total_time(log_file_path):
    with open(log_file_path, 'r') as file:
        for line in file:
            if 'Total time:' in line:
                match = re.search(r'Total time:\s*(\d{2}:\d{2}:\d{2})', line)
                if match:
                    time = match.group(1)
                    # converto to hrs from hh:mm:ss to float hours. Example '01:30:00' to 1.5
                    hours, minutes, seconds = map(int, time.split(':'))
                    total_hours = hours + minutes / 60 + seconds / 3600
                    return f'{total_hours:.2f}'
    return 0  # default if not found

def pointlossunidirectional(ptsa, ptsb):
    """point loss between ptsa, and nearest in ptsb
    N 2"""
    if type(ptsa) == torch.Tensor:
        ptsa = ptsa.cpu().numpy()
    if type(ptsb) == torch.Tensor:
        ptsb = ptsb.cpu().numpy()
   
    tree2 = KDTree(ptsb, leafsize=10)
    distances2, indices = tree2.query(ptsa)  # find closest to ptsa
    av_dist2 = np.mean(distances2)  # average euclidean distance (L2)

    distances1 = [sum(abs(ptsb[ind] - ptsa[i]) )for i, ind in enumerate(indices)]
    av_dist1 = np.mean(distances1)  # average L1

    return {
        "indices": indices.tolist(),
        "avgL1": av_dist1,
        "avgL2": av_dist2,
        "L1": distances1,
        "L2": distances2.tolist(),
    }

### Metrics functions

def img2mse(x, y, reduction='mean'):
    diff = torch.mean((x - y) ** 2, -1)
    if reduction == 'mean':
        return torch.mean(diff)
    elif reduction == 'sum':
        return torch.sum(diff)
    elif reduction == 'none':
        return diff

def mse2psnr(x):
    if isinstance(x, float):
        x = torch.tensor([x])
    return -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class ssim_utils:
    @staticmethod
    def ssim(img1, img2, window_size = 11, size_average = True):
        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel)
        
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        
        return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim(img1, img2, window_size = 11, size_average = True, format='NCHW'):
    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    return ssim_utils.ssim(img1, img2, window_size, size_average)

def calculate_recon_metrics(experiment_dir):
    """Calculate reconstruction metrics"""
    rendered_dir = Path(experiment_dir) / 'vid'
    rendered_dir = sorted(list(rendered_dir.glob('rend_right_*')))[-1]
    rendered_imgs = sorted(list(rendered_dir.glob('*.png')))
    if len(rendered_imgs) == 0:
        raise FileNotFoundError(f"No rendered images found in {rendered_dir}. Please check the path or generate the rendered images.")
    gt_dir = Path(experiment_dir).parent.parent / 'color_r'
    gt_imgs = [gt_dir / f.name for f in rendered_imgs]
    if not all(f.exists() for f in gt_imgs):
        raise FileNotFoundError(f"GT images not found in {gt_dir}. Please check the path or generate the GT images.")
    # Load images
    rendered_imgs = [torch.tensor(imageio.v2.imread(f)) for f in rendered_imgs]
    gt_imgs = [torch.tensor(imageio.v2.imread(f)) for f in gt_imgs]
    rendered_imgs = torch.stack(rendered_imgs, dim=0).float() / 255.0
    gt_imgs = torch.stack(gt_imgs, dim=0).float() / 255.0
    if rendered_imgs.shape != gt_imgs.shape:
        raise ValueError(f"Number of rendered images {rendered_imgs.shape} does not match number of GT images {gt_imgs.shape}.")

    mse = img2mse(rendered_imgs, gt_imgs)
    psnr = mse2psnr(mse)
    ssim_ = ssim(rendered_imgs, gt_imgs, format='NHWC')

    metrics = {
        "mse": mse.item(),
        "psnr": psnr.item(),
        "ssim": ssim_.item()
    }
    return metrics

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate tracker on each video os the STIR dataset')
    parser.add_argument('--vid_dir', '-v', type=str, help='Path to video directory')
    parser.add_argument('--out_dir', '-o', type=str, help='Path to output directory')
    parser.add_argument('--experiment', '-e', type=int, default=0, help='Experiment number')
    parser.add_argument('--vid_type', type=str, default='unknown', help='Type of video to evaluate (short, medium, long)')
    args = parser.parse_args()
    vid_dir, out_dir = Path(args.vid_dir), Path(args.out_dir)
    vid_dir = vid_dir.relative_to(vid_dir.parents[5]).parent.parent
    queries_file = out_dir.parent / 'gt_positions_start_all_test.json'
    gt_3DA_file = out_dir.parent / 'gt_3d_positions_start_all_test.json'
    gt_3DB_file = out_dir.parent / 'gt_3d_positions_end_all_test.json'
    gt_2DB_file = out_dir.parent / 'gt_positions_end_all_test.json'
    for file in [queries_file, gt_3DA_file, gt_3DB_file, gt_2DB_file]:
        if not file.exists():
            raise FileNotFoundError(f"Queries file {file} not found. Please generate it first.")
    with open(queries_file, 'r') as f:
        annotations = json.load(f)
    with open(gt_3DA_file, 'r') as f:
        gt_3DA_anno = json.load(f)
    with open(gt_3DB_file, 'r') as f:
        gt_3DB_anno = json.load(f)
    with open(gt_2DB_file, 'r') as f:
        gt_2DB_anno = json.load(f)

    start_marks, gt_3DA, gt_3DB, gt_2DB = [], [], [], []
    for key in annotations.keys():
        if str(vid_dir) in key:
            start_marks = np.array(annotations[key])
            gt_3DA = np.array(gt_3DA_anno[key])
            gt_3DB = np.array(gt_3DB_anno[key])
            gt_2DB = np.array(gt_2DB_anno[key])
            # print(f"Starting marks {start_marks.shape}: {start_marks}")
    for arr,file in zip([start_marks, gt_3DA, gt_3DB, gt_2DB], 
                                    [queries_file, gt_3DA_file, gt_3DB_file, gt_2DB_file]):
        if len(arr) == 0:
            raise ValueError(f"Video directory {vid_dir} not found in file {file} annotations. Please check the path or generate the file.")

    ## Evaluate all model configurations available
    failed = []
    for experiment_dir in (out_dir/'out').glob(f'Test_{args.experiment}*'):
        print(f"\nEvaluating tracker on {str(experiment_dir)}")
        try:
            predptsB, pred3DptsA, pred3DptsB, config = run_tracker(start_marks, experiment_dir)
            recon_metrics = calculate_recon_metrics(experiment_dir)
            exp_stem = experiment_dir.stem.split('_')
            log_file = experiment_dir / 'log_screen.txt'
            data = {
                "Experiment": exp_stem[0]+' '+exp_stem[1],
                "Date (aa-mm-dd)": exp_stem[2],
                "Starting time(hrs-min-sec)": exp_stem[3].replace('-', ':'),
                "Training time(hrs)": extract_total_time(log_file),
                "Video Path": str(vid_dir),
                "Video Type": args.vid_type,
                "qry 2DptsA": start_marks.tolist(),
                "Hat 3DptsA": pred3DptsA.tolist(),
                "Hat 3DptsB": pred3DptsB.tolist(),
                "Hat 2DptsB": predptsB.tolist(),
                "gt 3DptsA": gt_3DA.tolist(),
                "gt 3DptsB": gt_3DB.tolist(),
                "gt 2DptsB": gt_2DB.tolist(),
                # "error 3DptsA": pointlossunidirectional(pred3DptsA, gt_3DA),
                # "error 3DptsB": pointlossunidirectional(pred3DptsB, gt_3DB),
                # "error 2DptsB": pointlossunidirectional(predptsB, gt_2DB),
                "error 3DptsA": pointlossunidirectional(gt_3DA, pred3DptsA),
                "error 3DptsB": pointlossunidirectional(gt_3DB, pred3DptsB),
                "error 2DptsB": pointlossunidirectional(gt_2DB, predptsB),
                "recon mse": recon_metrics['mse'],
                "recon psnr": recon_metrics['psnr'],
                "recon ssim": recon_metrics['ssim'],
                "Notes": config.name}
            out_file = experiment_dir / 'evaluation.json'
            with open(out_file, 'w') as f:
                json.dump(data, f)
                
        except Exception as e:
            print(f"Error running tracker in folder {experiment_dir}")
            log_file = open(experiment_dir / 'eval_error_log.txt', 'w')
            log_file.write(traceback.format_exc())
            log_file.close()
            failed.append(experiment_dir)
            continue
    if failed:
        print(f"Evaluation failed for {len(failed)} fit(s), see eval_error_log.txt in: " + ", ".join(str(d) for d in failed))
        sys.exit(1)