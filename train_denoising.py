import torch
import torch.nn.functional as F
import time
import argparse
from torch.utils.data import DataLoader

from multiscale.gradients import ms_decompose
from multiscale.models import UNetModel, resnet
from multiscale.datasets import get_cifar10_dataset, get_celeba_dataset
from multiscale.utils import (
    set_seed,
    DynamicBatchData,
    setup_logging,
    save_logs,
    save_best_model,
    get_iteration_schedule
)

def val_loss(net, dataloader):
    device = next(net.parameters()).device
    data_iter = iter(dataloader)
    net.eval()
    batch = next(data_iter)
    x, _ = batch  # Unpack data and target
    x = x.to(device)
    z = torch.randn_like(x)
    t = torch.rand(x.shape[0], 1, 1, 1).to(device)
    xt = t * x + (1 - t) * z
    with torch.no_grad():
        x_hat = net(xt, t.squeeze())
        loss = F.mse_loss(x_hat, x)
    return loss

def train_multiscale_loss_denoising(net, data_loader, levels=3, batch_size0=8, scale=1):
    s = 1
    loss = 0
    device = next(net.parameters()).device
    data_loader.dataset.batch_size_function.current_size = batch_size0
    batch_size = data_loader.dataset.batch_size_function.current_size
    batch_size0 = batch_size
    for i, (data, target) in enumerate(data_loader):
        data_loader.dataset.batch_size_function.current_size = batch_size * 2
        x = data.to(device)
        z = torch.randn_like(x)
        t = torch.rand(x.shape[0], 1, 1, 1).to(device)
        xt = t * x + (1 - t) * z
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
            s = s / 2
            batch_size = batch_size * 2
        elif i == levels:
            xtF = F.interpolate(xt, scale_factor=s)
            xF = F.interpolate(x, scale_factor=s)
            xF_hat = net(xtF, t.squeeze())
            lossh = F.mse_loss(xF_hat, xF)
            loss = loss + lossh
            break
        data_loader.dataset.batch_size_function.current_size = batch_size0
    return loss

def get_dataset(dataset_name='cifar10', split='train', img_size=128):
    """
    Load dataset based on name.
    
    Args:
        dataset_name: 'cifar10' or 'celeba'
        split: 'train' or 'test' (or 'valid' for celeba)
        img_size: target image size
    """
    if dataset_name == 'cifar10':
        dataset = get_cifar10_dataset(split=split, img_size=img_size)
    elif dataset_name == 'celeba':
        dataset = get_celeba_dataset(split=split, img_size=img_size)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multiscale', 'fullmultiscale'])
    parser.add_argument('--network', type=str, default='unet', choices=['unet', 'resnet'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'celeba'])
    parser.add_argument('--img_size', type=int, default=128, help='Image size for training')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    # Set seed
    set_seed(args.run_id)

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    
    # Model selection
    if args.network == 'unet':
        model = UNetModel(
            in_channels=3,
            model_channels=32,
            out_channels=3,
            num_res_blocks=1,
            dropout=0.5,
            channel_mult=(1, 2, 4),
            attention_resolutions=[])
    elif args.network == 'resnet':
        model = resnet(
            in_channels=3,
            out_channels=3,
            hid_channels=128,
            nlayers=2,
            time_embed=True)
    else:
        raise ValueError(f'Unknown network: {args.network}')
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{args.network}: Total trainable params: {n_params:,}')
    
    model.to(device)
    
    # Load datasets
    train_dataset = get_dataset(dataset_name=args.dataset, split='train', img_size=args.img_size)
    test_dataset = get_dataset(dataset_name=args.dataset, split='test', img_size=args.img_size)
    
    # Create dynamic batch dataset
    dynamic_batch_size_obj = type('DynamicBatchSize', (), {'current_size': args.batch_size})()
    dynamic_train_dataset = DynamicBatchData(train_dataset, dynamic_batch_size_obj)
    train_loader = DataLoader(dynamic_train_dataset, batch_size=None, shuffle=False, num_workers=4)
    valid_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    niter_list = get_iteration_schedule(args.mode)
    T_max = sum(niter_list)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)

    # Setup logging
    exp_code = f'denoising_{args.dataset}_{args.network}_{args.mode}'
    logs_dir, models_dir, trainlog_fpath, validlog_fpath = setup_logging(
        exp_code=exp_code,
        run_id=args.run_id,
        main_dir='./results'
    )

    scale = 1
    TrainLossIter = []
    ValidLossIter = []
    iter_times = []
    best_val_loss = float('inf')
    total_start_time = time.perf_counter()
    
    # Print training configuration
    print('='*60)
    print(f'Starting training:')
    print(f'  Dataset: {args.dataset.upper()}')
    print(f'  Network: {args.network.upper()}')
    print(f'  Mode: {args.mode.upper()}')
    print(f'  Image size: {args.img_size}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Device: {device}')
    print(f'  Run ID: {args.run_id}')
    print('='*60)
    
    c = 0
    for j in range(4):
        f = 2 ** (3 - j)
        for i in range(niter_list[j]):
            iter_start_time = time.perf_counter()
            optimizer.zero_grad()
            if args.mode == 'fullmultiscale':
                loss = train_multiscale_loss_denoising(model, train_loader, levels=j, batch_size0=16 * f, scale=scale / f)
            elif args.mode == 'multiscale':
                loss = train_multiscale_loss_denoising(model, train_loader, levels=3, batch_size0=16, scale=1)
            elif args.mode == 'single':
                loss = train_multiscale_loss_denoising(model, train_loader, levels=0, batch_size0=16, scale=1)
            else:
                raise ValueError('Unknown mode')
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            iter_time = time.perf_counter() - iter_start_time
            iter_times.append(iter_time)
            TrainLossIter.append(loss.item())
            vloss = val_loss(model, valid_loader)
            ValidLossIter.append(vloss.item())
            
            # Save logs
            save_logs(trainlog_fpath, validlog_fpath, TrainLossIter, ValidLossIter, iter_times)
            
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print with level information for multiscale modes
            if args.mode in ['multiscale', 'fullmultiscale']:
                print(f'run_id:{args.run_id} | level={j}, iter={c}, trainloss={loss:.6f}, validloss={vloss:.6f}, lr={current_lr:.6f}')
            else:
                print(f'run_id:{args.run_id} | iter={c}, trainloss={loss:.6f}, validloss={vloss:.6f}, lr={current_lr:.6f}')
            
            # Save best model only
            if vloss.item() < best_val_loss:
                best_val_loss = vloss.item()
                save_best_model(models_dir, model, optimizer, j, c, loss.item(), vloss.item())
                print(f'  >>> Best model saved! Val loss: {vloss:.6f}')
            
            c += 1
    
    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    avg_iter_time = sum(iter_times) / len(iter_times)
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average time per iteration: {avg_iter_time:.4f} seconds")

if __name__ == '__main__':
    main()
