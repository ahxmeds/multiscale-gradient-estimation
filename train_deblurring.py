import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from multiscale.models import resnet, UNetModel
from multiscale.datasets import get_cifar10_dataset, get_celeba_dataset, get_stl10_dataset
from multiscale.forward_operators import blurFFT
from multiscale.utils import (
    set_seed,
    DynamicBatchData,
    setup_logging,
    get_iteration_schedule
)

def train_multiscale_loss_deblurring(net, data_loader, FP, levels, batch_size0, scale, device):
    """Multiscale loss computation for deblurring."""
    s = 1
    loss = 0
    
    data_loader.dataset.batch_size_function.current_size = batch_size0
    batch_size = data_loader.dataset.batch_size_function.current_size
    batch_size0 = batch_size
    for i, (data, target) in enumerate(data_loader):
        data_loader.dataset.batch_size_function.current_size = batch_size * 2
        x = data.to(device)
        z = torch.randn_like(x)
        t = 0.1 * torch.rand((x.shape[0], 1, 1, 1), device=device)
        x_blur = FP(x)
        xt = (1 - t) * x_blur + t * z
        
        x = F.interpolate(x, scale_factor=scale)
        xt = F.interpolate(xt, scale_factor=scale)
        if i < levels:
            xF = F.interpolate(x, scale_factor=s)
            xC = F.interpolate(xF, scale_factor=0.5)
            
            xtF = F.interpolate(xt, scale_factor=s)
            xtC = F.interpolate(xtF, scale_factor=0.5)
            
            xF_hat = net(xtF, t.squeeze())
            xC_hat = net(xtC, t.squeeze())
            
            lossh = F.mse_loss(xF_hat, xF)
            lossH = F.mse_loss(xC_hat, xC)
            dloss = lossh - lossH
            loss = loss + dloss
            
            print('level = %3d   bs = %3d  nH=%3d   nh=%3d   lH=%3.2e  lh=%3.2e   dloss=%3.2e'%(i, batch_size, xC.shape[-1], xF.shape[-1], lossH, lossh, dloss.abs()))
            s = s / 2
            batch_size = batch_size * 2

        elif i == levels:
            xtF = F.interpolate(xt, scale_factor=s)
            xF = F.interpolate(x, scale_factor=s)
            xF_hat = net(xtF, t.squeeze())
            lossh = F.mse_loss(xF_hat, xF)
            print('Coarsest Mesh: level=%3d   bs=%3d   nH=%3d   lh=%3.2e'%(i, batch_size, xF.shape[-1], lossh))

            loss = loss + lossh
            break
        data_loader.dataset.batch_size_function.current_size = batch_size0
    return loss

def val_loss(net, data_loader, FP, device):
    """Validation loss computation."""
    net.eval()
    data_iter = iter(data_loader)
    batch = next(data_iter)
    x = batch[0].to(device)
    x = F.interpolate(x, [128, 128])
    z = torch.randn_like(x)
    t = 0.1 * torch.rand((x.shape[0], 1, 1, 1), device=device)
    x_blur = FP(x)
    xt = (1 - t) * x_blur + t * z
    with torch.no_grad():
        x_hat = net(xt, t.squeeze())
        loss = F.mse_loss(x_hat, x)
    return loss

def main():
    parser = argparse.ArgumentParser(description='Train deblurring model with multiscale gradients')
    parser.add_argument('--dataset', type=str, default='stl10', choices=['cifar10', 'celeba', 'stl10'],
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
    exp_code = f'deblurring_{args.dataset}_{args.network}_{args.mode}'
    print(exp_code)
    
    logs_dir, models_dir, trainlog_fpath, validlog_fpath = setup_logging(
        exp_code=exp_code,
        run_id=args.run_id,
        main_dir='./results'
    )
    
    # Setup device
    device = torch.device(f'cuda:{args.device_id}')
    
    # Load datasets (no img_size parameter for STL10)
    if args.dataset == 'cifar10':
        train_dataset = get_cifar10_dataset(split='train')
        valid_dataset = get_cifar10_dataset(split='test')
    elif args.dataset == 'celeba':
        train_dataset = get_celeba_dataset(split='train')
        valid_dataset = get_celeba_dataset(split='valid')
    elif args.dataset == 'stl10':
        train_dataset = get_stl10_dataset(split='train')
        valid_dataset = get_stl10_dataset(split='test')
    
    # Create dynamic batch dataset with initial size 8
    dynamic_batch_size = type('DynamicBatchSize', (), {'current_size': 8})()
    dynamic_train_dataset = DynamicBatchData(train_dataset, dynamic_batch_size)
    
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
    
    # Initialize forward operator (blur)
    FP = blurFFT(dim=128, sigma=[3, 3], device=device)
    
    # Print training configuration
    print('='*60)
    print(f'Starting training:')
    print(f'  Dataset: {args.dataset.upper()}')
    print(f'  Network: {args.network.upper()}')
    print(f'  Mode: {args.mode.upper()}')
    print(f'  Device: {device}')
    print(f'  Run ID: {args.run_id}')
    print('='*60)
    
    # Training loop
    scale = 1
    c = 0
    best_val_loss = float('inf')
    TrainLossIter = []
    ValidLossIter = []
    
    for j in range(4):
        f = 2 ** (3 - j)
        for i in range(niter_list[j]):
            optimizer.zero_grad()
            
            if args.mode == 'fullmultiscale':  # FMG cycle
                loss = train_multiscale_loss_deblurring(model, train_loader, FP, levels=j, 
                                                       batch_size0=16*f, scale=scale/f, device=device)
            elif args.mode == 'multiscale':
                loss = train_multiscale_loss_deblurring(model, train_loader, FP, levels=3, 
                                                       batch_size0=16, scale=1, device=device)
            elif args.mode == 'single':  # Fine mesh only
                loss = train_multiscale_loss_deblurring(model, train_loader, FP, levels=0, 
                                                       batch_size0=16, scale=1, device=device)
            else:
                print('Incorrect methodology given!!!')
            
            loss.backward()
            optimizer.step()
            
            TrainLossIter.append(loss.item())
            
            # Validation
            vloss = val_loss(model, valid_loader, FP, device)
            ValidLossIter.append(vloss.item())
            
            # Save logs
            train_df = pd.DataFrame(TrainLossIter, columns=['Loss'])
            valid_df = pd.DataFrame(ValidLossIter, columns=['Loss'])
            train_df.to_csv(trainlog_fpath, index=False)
            valid_df.to_csv(validlog_fpath, index=False)
            
            scheduler.step()
            
            # Print progress
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print with level information for multiscale modes
            if args.mode in ['multiscale', 'fullmultiscale']:
                print(f'run_id:{args.run_id} | level={j}, iter={c}, trainloss={loss:.6f}, validloss={vloss:.6f}, lr={current_lr:.6f}')
            else:
                print(f'run_id:{args.run_id} | iter={c}, trainloss={loss:.6f}, validloss={vloss:.6f}, lr={current_lr:.6f}')
            
            # Save best model only
            if vloss.item() < best_val_loss:
                best_val_loss = vloss.item()
                model_save_fpath = os.path.join(models_dir, 'best_model.pt')
                torch.save({
                    'epoch': j,
                    'iter': c,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': loss.item(),
                    'val_loss': vloss.item(),
                }, model_save_fpath)
                print(f'  >>> Best model saved! Val loss: {vloss:.6f}')
            
            c += 1
    
    torch.cuda.empty_cache()
    print(f'\\nTraining completed! Best validation loss: {best_val_loss:.6f}')
    print(f'Model saved to: {models_dir}')

if __name__ == '__main__':
    main()
