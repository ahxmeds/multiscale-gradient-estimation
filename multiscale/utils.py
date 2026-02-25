import torch
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import pandas as pd
from torch.utils.data import IterableDataset

def pad_zeros_at_front(num, N):
    """Add leading zeros to number for consistent file naming."""
    return str(num).zfill(N)

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class DynamicBatchData(IterableDataset):
    """Dynamic batch size dataset wrapper."""
    """This is used for deblurring and denoising where we only have one image per sample."""
    def __init__(self, dataset, batch_size_function):
        self.dataset = dataset
        self.batch_size_function = batch_size_function
        self.index = 0
    
    def __iter__(self):
        while True:
            current_batch_size = self.batch_size_function.current_size
            batch = []
            for _ in range(current_batch_size):
                if self.index >= len(self.dataset):
                    self.index = 0
                batch.append(self.dataset[self.index])
                self.index += 1
            yield self.collate_batch(batch)
    
    def collate_batch(self, batch):
        data, target = zip(*batch)
        data = torch.stack(data)
        data = F.interpolate(data, [128, 128])
        target = torch.tensor(target)
        return data, target

class DynamicBatchDataInpainting(IterableDataset):
    """Dynamic batch size dataset wrapper for inpainting (returns pairs)."""
    def __init__(self, dataset, batch_size_function):
        self.dataset = dataset
        self.batch_size_function = batch_size_function
        self.index = 0
    
    def __iter__(self):
        while True:
            current_batch_size = self.batch_size_function.current_size
            batch = []
            for _ in range(current_batch_size):
                if self.index >= len(self.dataset):
                    self.index = 0
                batch.append(self.dataset[self.index])
                self.index += 1
            yield self.collate_batch(batch)
    
    def collate_batch(self, batch):
        data, target = zip(*batch)
        data = torch.stack(data)
        target = torch.stack(target)
        data = F.interpolate(data, [128, 128])
        target = F.interpolate(target, [128, 128])
        return data, target

class DynamicBatchDataSuperResolution(IterableDataset):
    """
    Dynamic batch size dataset wrapper for super-resolution.
    Returns (low_res, high_res) image pairs.
    """
    def __init__(self, dataset, batch_size_function):
        self.dataset = dataset
        self.batch_size_function = batch_size_function
        self.index = 0
    
    def __iter__(self):
        while True:
            current_batch_size = self.batch_size_function.current_size
            batch = []
            for _ in range(current_batch_size):
                if self.index >= len(self.dataset):
                    self.index = 0
                batch.append(self.dataset[self.index])
                self.index += 1
            yield self.collate_batch(batch)
    
    def collate_batch(self, batch):
        """Collate low-res and high-res image pairs."""
        lr_images, hr_images = zip(*batch)
        lr_images = torch.stack(lr_images)
        hr_images = torch.stack(hr_images)
        return lr_images, hr_images


def dynamic_batch_size():
    """Return current batch size."""
    return dynamic_batch_size.current_size

def setup_logging(exp_code, run_id, main_dir='./results'):
    """Setup logging directories and return paths."""
    logs_dir = os.path.join(main_dir, 'logs', f'run{run_id}', exp_code)
    models_dir = os.path.join(main_dir, 'models', f'run{run_id}', exp_code)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    trainlog_fpath = os.path.join(logs_dir, 'train_logs.csv')
    validlog_fpath = os.path.join(logs_dir, 'valid_logs.csv')
    return logs_dir, models_dir, trainlog_fpath, validlog_fpath

def save_logs(trainlog_fpath, validlog_fpath, TrainLossIter, ValidLossIter, iter_times):
    """Save training and validation logs to CSV."""
    train_df = pd.DataFrame(TrainLossIter, columns=['Loss'])
    valid_df = pd.DataFrame(np.column_stack((ValidLossIter, iter_times)), columns=['Loss', 'IterTime'])
    train_df.to_csv(trainlog_fpath, index=False)
    valid_df.to_csv(validlog_fpath, index=False)

def save_best_model(models_dir, model, optimizer, epoch, iteration, train_loss, val_loss):
    """Save best model checkpoint."""
    model_save_fpath = os.path.join(models_dir, 'best_model.pt')
    torch.save({
        'epoch': epoch,
        'iter': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, model_save_fpath)

def print_training_header(args, device, n_params):
    """Print training configuration header."""
    print('='*60)
    print(f'Starting training:')
    print(f'  Dataset: {args.dataset.upper()}')
    print(f'  Network: {args.network.upper()}')
    print(f'  Mode: {args.mode.upper()}')
    print(f'  Image size: {args.img_size}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Device: {device}')
    print(f'  Run ID: {args.run_id}')
    print(f'  Total params: {n_params:,}')
    print('='*60)

def get_iteration_schedule(mode):
    """Get iteration schedule based on training mode."""
    if mode == 'single':
        niter_list = [2000, 2000, 2000, 2000]
    elif mode == 'multiscale':
        niter_list = [2000, 2000, 2000, 2000]
    elif mode == 'fullmultiscale':
        niter_list = [2000, 1000, 500, 250]
    else:
        raise ValueError(f'Unknown mode: {mode}')
    return niter_list

def gaussian_kernel(window_size, sigma, channels):
    """Create Gaussian kernel for SSIM computation."""
    gauss = torch.arange(window_size).float()
    gauss = torch.exp(-((gauss - window_size // 2) ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    kernel = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    kernel = kernel.expand(channels, 1, window_size, window_size)
    return kernel

def compute_ssim(img1, img2, window_size=11, sigma=1.5, L=1.0):
    """Compute Structural Similarity Index (SSIM) between two images."""
    channels = img1.size(1)
    kernel = gaussian_kernel(window_size, sigma, channels).to(img1.device)
    
    mu1 = F.conv2d(img1, kernel, groups=channels, padding=window_size//2)
    mu2 = F.conv2d(img2, kernel, groups=channels, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 ** 2, kernel, groups=channels, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, groups=channels, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, groups=channels, padding=window_size//2) - mu1_mu2
    
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean([1, 2, 3])  # Mean over spatial dimensions and channels


