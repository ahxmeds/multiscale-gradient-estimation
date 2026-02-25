import os
import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from multiscale.models import resnet, SRParaConvNet
from multiscale.datasets import get_urban100_dataset
from multiscale.utils import (
    set_seed,
    DynamicBatchDataSuperResolution,
    setup_logging,
    save_logs,
    save_best_model,
    get_iteration_schedule,
    compute_ssim
)


def compute_sr_loss(model, lr_images, hr_images, device):
    """
    Compute super-resolution loss using negative SSIM.
    
    Args:
        model: Super-resolution network
        lr_images: Low-resolution input images
        hr_images: High-resolution target images
        device: Device to run on
        
    Returns:
        Negative SSIM loss (to be minimized)
    """
    # Bilinear upsampling baseline (2x)
    upsampled = F.interpolate(lr_images, scale_factor=2, mode='bilinear', align_corners=False)
    
    # Model refinement (learns residual)
    residual = model(upsampled)
    output = residual + upsampled
    
    # SSIM loss (negative because we want to maximize SSIM)
    ssim = compute_ssim(output, hr_images, window_size=11, sigma=1.5, L=1.0)
    loss = -ssim.mean()
    
    return loss


def train_multiscale_loss_superresolution(model, data_loader, levels, batch_size0, scale, device):
    """
    Multiscale loss computation for super-resolution.
    
    This function implements a multiscale training strategy where the loss is
    computed at multiple resolution levels. At coarser levels, larger batches
    are used to compensate for the reduced computational cost.
    
    Args:
        model: Super-resolution network
        data_loader: Training data loader
        levels: Number of resolution levels
        batch_size0: Starting batch size at finest level
        scale: Global scaling factor for input resolution
        device: Device to run on
        
    Returns:
        Accumulated multiscale loss
    """
    model.train()
    s = 1
    loss = 0
    
    data_loader.dataset.batch_size_function.current_size = batch_size0
    batch_size = batch_size0
    
    for i, (lr_images, hr_images) in enumerate(data_loader):
        data_loader.dataset.batch_size_function.current_size = batch_size * 2
        
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        
        # Apply global scale
        lr_images = F.interpolate(lr_images, scale_factor=scale, mode='bilinear', align_corners=False)
        hr_images = F.interpolate(hr_images, scale_factor=scale, mode='bilinear', align_corners=False)
        
        if i < levels:
            # Fine scale
            xF = F.interpolate(lr_images, scale_factor=s, mode='bilinear', align_corners=False)
            xtF = F.interpolate(hr_images, scale_factor=s, mode='bilinear', align_corners=False)
            
            # Coarse scale
            xC = F.interpolate(xF, scale_factor=0.5, mode='bilinear', align_corners=False)
            xtC = F.interpolate(xtF, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            # Compute losses at both scales
            lossh = compute_sr_loss(model, xF, xtF, device)
            lossH = compute_sr_loss(model, xC, xtC, device)
            dloss = lossh - lossH
            loss = loss + dloss
            
            print(f'level={i:3d}   bs={batch_size:3d}   nH={xC.shape[-1]:3d}   '
                  f'nh={xF.shape[-1]:3d}   lH={lossH:.2e}   lh={lossh:.2e}   dloss={dloss.abs():.2e}')
            
            s = s / 2
            batch_size = batch_size * 2
            
        elif i == levels:
            # Coarsest mesh
            xtF = F.interpolate(hr_images, scale_factor=s, mode='bilinear', align_corners=False)
            xF = F.interpolate(lr_images, scale_factor=s, mode='bilinear', align_corners=False)
            
            lossh = compute_sr_loss(model, xF, xtF, device)
            print(f'Coarsest Mesh: level={i:3d}   bs={batch_size:3d}   nH={xF.shape[-1]:3d}   lh={lossh:.2e}')
            
            loss = loss + lossh
            break
    
    data_loader.dataset.batch_size_function.current_size = batch_size0
    return loss


def val_loss(model, data_loader, device):
    """Validation loss computation."""
    model.eval()
    data_iter = iter(data_loader)
    batch = next(data_iter)
    
    lr_images = batch[0].to(device)
    hr_images = batch[1].to(device)
    
    with torch.no_grad():
        loss = compute_sr_loss(model, lr_images, hr_images, device)
    
    return loss


def main():
    parser = argparse.ArgumentParser(description='Train super-resolution model with multiscale gradients')
    parser.add_argument('--dataset', type=str, default='urban100', choices=['urban100'],
                        help='Dataset to use (currently only Urban100 supported)')
    parser.add_argument('--network', type=str, default='srnet', choices=['srnet', 'resnet'],
                        help='Network architecture')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multiscale', 'fullmultiscale'],
                        help='Training mode: single (finemesh), multiscale (ms), fullmultiscale (fms)')
    parser.add_argument('--run_id', type=int, default=1,
                        help='Run ID for experiment tracking')
    parser.add_argument('--device_id', type=int, default=0,
                        help='CUDA device ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Patch size for training')
    parser.add_argument('--train_patches', type=int, default=10,
                        help='Number of patches per training image')
    parser.add_argument('--test_patches', type=int, default=2,
                        help='Number of patches per test image')
    
    args = parser.parse_args()
    
    # Set seed and device
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment code
    mode_short = {'single': 'single', 'multiscale': 'ms', 'fullmultiscale': 'fms'}[args.mode]
    exp_code = f'superresolution_{args.dataset}_{args.network}_{mode_short}_run{args.run_id}'
    
    # Setup logging
    logs_dir, models_dir, trainlog_fpath, validlog_fpath = setup_logging(exp_code, args.run_id)
    
    print('\n' + '='*70)
    print(f'Experiment: {exp_code}')
    print('='*70 + '\n')
    
    # Load datasets
    print('Loading datasets...')
    train_dataset = get_urban100_dataset(
        split='train',
        num_patches=args.train_patches,
        patch_size=args.patch_size
    )
    valid_dataset = get_urban100_dataset(
        split='test',
        num_patches=args.test_patches,
        patch_size=args.patch_size
    )
    
    # Dynamic batch size function
    def dynamic_batch_size():
        return dynamic_batch_size.current_size
    dynamic_batch_size.current_size = 8
    
    # Create data loaders
    dynamic_train_dataset = DynamicBatchDataSuperResolution(train_dataset, dynamic_batch_size)
    train_loader = DataLoader(dynamic_train_dataset, batch_size=None, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=200, shuffle=False, num_workers=4)
    
    # Initialize model
    if args.network == 'srnet':
        model = SRParaConvNet(
            in_channels=3,
            out_channels=3,
            hid_channels=64,
            upscale_factor=2
        )
    elif args.network == 'resnet':
        model = resnet(
            in_channels=3,
            out_channels=3,
            hid_channels=100,
            nlayers=9,
            time_embed=False
        )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n{args.network}: Total trainable params: {n_params:,}\n')
    
    # Move model to device
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    niter_list = get_iteration_schedule(args.mode)
    T_max = sum(niter_list)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    
    # Training loop
    scale = 1
    c = 0
    best_val_loss = float('inf')
    TrainLossIter = []
    ValidLossIter = []
    iter_times = []
    
    print('='*70)
    print(f'Starting training:')
    print(f'  Dataset: {args.dataset.upper()}')
    print(f'  Network: {args.network.upper()}')
    print(f'  Mode: {args.mode.upper()}')
    print(f'  Patch size: {args.patch_size}')
    print(f'  Device: {device}')
    print(f'  Run ID: {args.run_id}')
    print('='*70 + '\n')
    
    for j in range(4):
        f = 2 ** (3 - j)
        for i in range(niter_list[j]):
            iter_start_time = time.perf_counter()
            optimizer.zero_grad()
            
            if args.mode == 'fullmultiscale':  # FMG cycle
                loss = train_multiscale_loss_superresolution(model, train_loader, levels=j,
                                                batch_size0=8*f, scale=scale/f, device=device)
            elif args.mode == 'multiscale':
                loss = train_multiscale_loss_superresolution(model, train_loader, levels=3,
                                                batch_size0=8, scale=1, device=device)
            elif args.mode == 'single':  # Fine mesh only
                loss = train_multiscale_loss_superresolution(model, train_loader, levels=0,
                                                batch_size0=8, scale=1, device=device)
            
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            iter_time = time.perf_counter() - iter_start_time
            iter_times.append(iter_time)
            
            TrainLossIter.append(loss.item())
            
            # Validation
            vloss = val_loss(model, valid_loader, device)
            ValidLossIter.append(vloss.item())
            
            # Save logs
            save_logs(trainlog_fpath, validlog_fpath, TrainLossIter, ValidLossIter, iter_times)
            
            # Print progress
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print with level information for multiscale modes
            if args.mode in ['multiscale', 'fullmultiscale']:
                print(f'run_id:{args.run_id} | level={j}, iter={c}, trainloss={loss.item():.6f}, '
                      f'validloss={vloss.item():.6f}, lr={current_lr:.6f}')
            else:
                print(f'run_id:{args.run_id} | iter={c}, trainloss={loss.item():.6f}, '
                      f'validloss={vloss.item():.6f}, lr={current_lr:.6f}')
            
            # Save best model only
            if vloss.item() < best_val_loss:
                best_val_loss = vloss.item()
                save_best_model(models_dir, model, optimizer, j, c, loss.item(), vloss.item())
                print(f'  >>> Best model saved! Val loss: {vloss.item():.6f}')
            
            scheduler.step()
            c += 1
    
    print(f'\nTraining completed! Best validation loss: {best_val_loss:.6f}')
    print(f'Model saved to: {models_dir}')

if __name__ == '__main__':
    main()
