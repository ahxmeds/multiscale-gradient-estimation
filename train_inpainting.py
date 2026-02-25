import os
import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from multiscale.models import resnet, UNetModel
from multiscale.datasets import get_cifar10_inpainting_dataset, get_celeba_inpainting_dataset
from multiscale.utils import (
    set_seed,
    DynamicBatchDataInpainting,
    setup_logging,
    save_logs,
    save_best_model,
    print_training_header,
    get_iteration_schedule,
    compute_ssim
)

def train_multiscale_loss_inpainting(net, data_loader, levels, batch_size0, scale, device):
    """Multiscale loss computation for inpainting."""
    net.train()
    batch = next(iter(data_loader))
    x_cor, x_tar = batch[0].to(device), batch[1].to(device)
    
    # Sample random timestep (0.1 multiplier for inpainting)
    t = 0.1 * torch.rand((x_cor.shape[0], 1, 1, 1), device=device)
    z = torch.randn_like(x_cor)
    xt = (1 - t) * x_cor + t * z  # Corrupt image with noise
    
    # Scale the images
    x_tar = F.interpolate(x_tar, scale_factor=scale)
    xt = F.interpolate(xt, scale_factor=scale)
    
    loss = 0
    s = 1
    batch_size = batch_size0
    data_loader.dataset.batch_size_function.current_size = batch_size
    
    for i in range(levels + 1):
        if i < levels:
            xtar_F = F.interpolate(x_tar, scale_factor=s)
            xtar_C = F.interpolate(xtar_F, scale_factor=0.5)
            
            xtF = F.interpolate(xt, scale_factor=s)
            xtC = F.interpolate(xtF, scale_factor=0.5)
            
            xF_hat = net(xtF, t.squeeze())
            xC_hat = net(xtC, t.squeeze())
            
            lossh = F.mse_loss(xF_hat, xtar_F)
            lossH = F.mse_loss(xC_hat, xtar_C)
            dloss = lossh - lossH
            loss = loss + dloss
            
            print(f'level={i:3d}   bs={batch_size:3d}  nH={xtar_C.shape[-1]:3d}   nh={xtar_F.shape[-1]:3d}   '
                  f'lH={lossH:.2e}  lh={lossh:.2e}   dloss={dloss.abs():.2e}')
            s = s / 2
            batch_size = batch_size * 2
            
        elif i == levels:
            xtF = F.interpolate(xt, scale_factor=s)
            xtar_F = F.interpolate(x_tar, scale_factor=s)
            xF_hat = net(xtF, t.squeeze())
            lossh = F.mse_loss(xF_hat, xtar_F)
            print(f'Coarsest Mesh: level={i:3d}   bs={batch_size:3d}   nH={xtar_F.shape[-1]:3d}   lh={lossh:.2e}')
            
            loss = loss + lossh
            break
            
    data_loader.dataset.batch_size_function.current_size = batch_size0
    return loss

def val_loss(net, data_loader, device):
    """Validation loss computation with SSIM metric."""
    net.eval()
    total_loss = 0
    total_ssim = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            x_cor = batch[0].to(device)
            x_cor = F.interpolate(x_cor, [128, 128])
            x_tar = batch[1].to(device)
            x_tar = F.interpolate(x_tar, [128, 128])
            
            z = torch.randn_like(x_cor)
            t = 0.1 * torch.rand((x_cor.shape[0], 1, 1, 1), device=device)
            xt = (1 - t) * x_cor + t * z
            
            x_hat = net(xt, t.squeeze())
            loss = F.mse_loss(x_hat, x_tar)
            ssim = torch.mean(compute_ssim(x_hat, x_tar))
            
            total_loss += loss.item()
            total_ssim += ssim.item()
            num_batches += 1
            
    return total_loss / num_batches, total_ssim / num_batches

def main():
    parser = argparse.ArgumentParser(description='Train inpainting model with multiscale gradients')
    parser.add_argument('--dataset', type=str, default='celeba', choices=['cifar10', 'celeba'],
                      help='Dataset to use')
    parser.add_argument('--network', type=str, default='unet', choices=['unet', 'resnet'],
                      help='Network architecture')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multiscale', 'fullmultiscale'],
                      help='Training mode: single (finemesh), multiscale (ms), fullmultiscale (fms)')
    parser.add_argument('--run_id', type=int, default=1,
                      help='Run ID for experiment tracking')
    parser.add_argument('--device_id', type=int, default=0,
                      help='CUDA device ID')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup experiment code and directories
    exp_code = f'inpainting_{args.dataset}_{args.network}_{args.mode}'
    print(f'Experiment: {exp_code}')
    
    logs_dir, models_dir, trainlog_fpath, validlog_fpath = setup_logging(
        exp_code=exp_code,
        run_id=args.run_id,
        main_dir='./results'
    )
    
    # Setup device
    device = torch.device(f'cuda:{args.device_id}')
    
    # Load datasets - for inpainting, we need corrupted/target pairs
    # Note: This requires datasets that provide pairs of (corrupted, target) images
    if args.dataset == 'cifar10':
        train_dataset = get_cifar10_inpainting_dataset(split='train')
        valid_dataset = get_cifar10_inpainting_dataset(split='test')
    elif args.dataset == 'celeba':
        train_dataset = get_celeba_inpainting_dataset(split='train')
        valid_dataset = get_celeba_inpainting_dataset(split='valid')
    
    # Create dynamic batch dataset for inpainting (returns pairs)
    dynamic_batch_size = type('DynamicBatchSize', (), {'current_size': 16})()
    dynamic_train_dataset = DynamicBatchDataInpainting(train_dataset, dynamic_batch_size)
    
    # Create data loaders
    train_loader = DataLoader(dynamic_train_dataset, batch_size=None, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    # Initialize model
    if args.network == 'unet':
        model = UNetModel(
            in_channels=3,
            model_channels=32,
            out_channels=3,
            num_res_blocks=1,
            dropout=0.5,
            channel_mult=(1, 2, 4),
            attention_resolutions=[]
        )
    elif args.network == 'resnet':
        model = resnet(
            in_channels=3,
            out_channels=3,
            hid_channels=128,
            nlayers=2,
            time_embed=True
        )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n{args.network}: Total trainable params: {n_params:,}\n')
    
    # Move model to device
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    niter_list = get_iteration_schedule(args.mode)
    T_max = sum(niter_list)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    
    # Training loop
    scale = 1
    c = 0
    best_val_loss = float('inf')
    TrainLossIter = []
    ValidLossIter = []
    ValidSSIMIter = []
    iter_times = []
    
    for j in range(4):
        f = 2 ** (3 - j)
        for i in range(niter_list[j]):
            iter_start_time = time.perf_counter()
            optimizer.zero_grad()
            
            if args.mode == 'fullmultiscale':  # FMG cycle
                loss = train_multiscale_loss_inpainting(model, train_loader, levels=j, 
                                                       batch_size0=16*f, scale=scale/f, device=device)
            elif args.mode == 'multiscale':
                loss = train_multiscale_loss_inpainting(model, train_loader, levels=3, 
                                                       batch_size0=16, scale=1, device=device)
            elif args.mode == 'single':  # Fine mesh only
                loss = train_multiscale_loss_inpainting(model, train_loader, levels=0, 
                                                       batch_size0=16, scale=1, device=device)
            
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            iter_time = time.perf_counter() - iter_start_time
            iter_times.append(iter_time)
            
            TrainLossIter.append(loss.item())
            
            # Validation
            vloss, vssim = val_loss(model, valid_loader, device)
            ValidLossIter.append(vloss)
            ValidSSIMIter.append(vssim)
            
            # Save logs
            save_logs(trainlog_fpath, validlog_fpath, TrainLossIter, ValidLossIter, iter_times)
            
            # Save best model
            if vloss < best_val_loss:
                best_val_loss = vloss
                save_best_model(models_dir, model, optimizer, j, c, loss.item(), vloss)
            
            # Print progress
            current_lr = optimizer.param_groups[0]['lr']
            print('####################################')
            print(f'run_id:{args.run_id} | iter={c}, trainloss={loss.item():.6f}, '
                  f'validloss={vloss:.6f}, ssim={vssim:.6f}, lr={current_lr:.6f}')
            print('####################################')
            
            scheduler.step()
            c += 1
    
    print(f'\nTraining completed! Best validation loss: {best_val_loss:.6f}')
    print(f'Model saved to: {models_dir}')

if __name__ == '__main__':
    main()
