import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset

MAIN_DATA_DIR = '/data/sahamed/DATA'

def get_cifar10_dataset(split='train', img_size=None):
    """
    Load CIFAR10 dataset.
    
    Args:
        split: 'train' or 'test'
        img_size: target image size (e.g., 128). If None, uses original size (32x32)
    """
    mean_ = [0.5, 0.5, 0.5]
    std_ = [0.5, 0.5, 0.5]
    
    if img_size is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize(mean=mean_, std=std_)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_, std=std_)
        ])
    
    if split == 'train':
        dataset = datasets.CIFAR10(root=MAIN_DATA_DIR, train=True, download=True, transform=transform)
    elif split == 'test':
        dataset = datasets.CIFAR10(root=MAIN_DATA_DIR, train=False, download=True, transform=transform)
    else:
        raise ValueError(f'Invalid split value: {split}')
    
    return dataset


class CIFAR10InpaintingWrapper(Dataset):
    """Wrapper for CIFAR10 that returns (image, image) pairs for inpainting."""
    def __init__(self, cifar10_dataset):
        self.dataset = cifar10_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Ignore label
        return image, image  # Return same image twice


def get_cifar10_inpainting_dataset(split='train', img_size=None):
    """
    Load CIFAR10 dataset for inpainting (returns image pairs).
    
    Args:
        split: 'train' or 'test'
        img_size: target image size (e.g., 128). If None, uses original size (32x32)
    """
    cifar10_dataset = get_cifar10_dataset(split=split, img_size=img_size)
    return CIFAR10InpaintingWrapper(cifar10_dataset)


def get_stl10_dataset(split='train', img_size=None):
    """
    Load STL10 dataset.
    
    Args:
        split: 'train' or 'test'
        img_size: target image size (e.g., 128). If None, uses original size (96x96)
    """
    if img_size is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2240, 0.2215, 0.2239])             
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2240, 0.2215, 0.2239])             
        ])

    if split == 'train':
        train_dataset = datasets.STL10(root=MAIN_DATA_DIR, split='train', download=False, transform=transform)
        unlabeled_dataset = datasets.STL10(root=MAIN_DATA_DIR, split='unlabeled', download=False, transform=transform)
        dataset = train_dataset + unlabeled_dataset 
    elif split == 'test':
        dataset = datasets.STL10(root=MAIN_DATA_DIR, split='test', download=False, transform=transform)
    else:
        raise ValueError(f'Invalid split value: {split}')
    
    return dataset


class CelebADataset(Dataset):
    """Custom CelebA Dataset loader."""
    def __init__(self, img_dir, attr_file, split_file, split='train', transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.attr_df = pd.read_csv(attr_file, sep=r'\s+', header=1)
        self.img_names = self.attr_df.index.values
        self.attrs = self.attr_df.values
        self.split_df = pd.read_csv(split_file, sep=r'\s+', header=None, index_col=0)

        if split == 'train':
            self.split_idx = self.split_df[self.split_df[1] == 0].index
        elif split == 'valid':
            self.split_idx = self.split_df[self.split_df[1] == 1].index
        elif split == 'test':
            self.split_idx = self.split_df[self.split_df[1] == 2].index
        else:
            raise ValueError("Split must be one of 'train', 'valid', or 'test'")

        self.img_names = self.split_idx.values
        self.attrs = self.attr_df.loc[self.img_names].values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        attributes = self.attrs[idx]
        return image, attributes


class CelebAInpaintingDataset(Dataset):
    """CelebA Dataset for inpainting - returns (image, image) pairs."""
    def __init__(self, img_dir, attr_file, split_file, split='train', transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.attr_df = pd.read_csv(attr_file, sep=r'\s+', header=1)
        self.img_names = self.attr_df.index.values
        self.split_df = pd.read_csv(split_file, sep=r'\s+', header=None, index_col=0)

        if split == 'train':
            self.split_idx = self.split_df[self.split_df[1] == 0].index
        elif split == 'valid':
            self.split_idx = self.split_df[self.split_df[1] == 1].index
        elif split == 'test':
            self.split_idx = self.split_df[self.split_df[1] == 2].index
        else:
            raise ValueError("Split must be one of 'train', 'valid', or 'test'")

        self.img_names = self.split_idx.values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        # For inpainting, return the same image twice (will be corrupted during training)
        return image, image


def get_celeba_dataset(split='train', img_size=64):
    """
    Load CelebA dataset.
    
    Args:
        split: 'train', 'valid', or 'test'
        img_size: target image size (e.g., 64, 128)
    """
    data_dir = os.path.join(MAIN_DATA_DIR, 'CelebA', 'img_align_celeba')
    attr_file = os.path.join(MAIN_DATA_DIR, 'CelebA', 'list_attr_celeba.txt')
    split_file = os.path.join(MAIN_DATA_DIR, 'CelebA', 'list_eval_partition.txt')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = CelebADataset(img_dir=data_dir, attr_file=attr_file, split_file=split_file, 
                           split=split, transform=transform)
    return dataset


def get_celeba_inpainting_dataset(split='train', img_size=64):
    """
    Load CelebA dataset for inpainting (returns image pairs).
    
    Args:
        split: 'train', 'valid', or 'test'
        img_size: target image size (e.g., 64, 128)
    """
    data_dir = os.path.join(MAIN_DATA_DIR, 'CelebA', 'img_align_celeba')
    attr_file = os.path.join(MAIN_DATA_DIR, 'CelebA', 'list_attr_celeba.txt')
    split_file = os.path.join(MAIN_DATA_DIR, 'CelebA', 'list_eval_partition.txt')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = CelebAInpaintingDataset(img_dir=data_dir, attr_file=attr_file, split_file=split_file, 
                                      split=split, transform=transform)
    return dataset


# ============================================================================
# Urban100 Dataset for Super-Resolution
# ============================================================================

class Urban100Dataset(Dataset):
    """
    Dataset class for Urban100 super-resolution image pairs.
    Contains pre-extracted patches of low-resolution and high-resolution images.
    
    Args:
        low_res_images: Tensor of low-resolution images/patches
        high_res_images: Tensor of corresponding high-resolution images/patches
        img_size: Target image size (not used, kept for consistency)
    """
    
    def __init__(
        self,
        low_res_images: torch.Tensor,
        high_res_images: torch.Tensor,
        img_size: int = 128
    ):
        self.low_res_images = low_res_images
        self.high_res_images = high_res_images
        self.img_size = img_size
    
    def __len__(self) -> int:
        return len(self.low_res_images)
    
    def __getitem__(self, idx: int):
        return self.low_res_images[idx], self.high_res_images[idx]


def generate_urban100_patches(
    low_res_path: str,
    high_res_path: str,
    num_patches: int = 100,
    patch_size: int = 64
):
    """
    Generate training/validation dataset by extracting random patches from Urban100 images.
    
    Args:
        low_res_path: Path to low-resolution images tensor file (.pt)
        high_res_path: Path to high-resolution images tensor file (.pt)
        num_patches: Number of random patches to extract per image
        patch_size: Size of patches to extract from low-res images
        
    Returns:
        Tuple of (low_res_patches, high_res_patches) tensors
    """
    import torch
    
    lr_images = torch.load(low_res_path)
    hr_images = torch.load(high_res_path)
    
    low_res_patches = []
    high_res_patches = []
    
    for lr_img, hr_img in zip(lr_images, hr_images):
        for _ in range(num_patches):
            # Random patch extraction
            x_offset = torch.randint(lr_img.shape[1] - patch_size, (1,)).item()
            y_offset = torch.randint(lr_img.shape[2] - patch_size, (1,)).item()
            
            # Extract corresponding patches (2x upscaling)
            lr_patch = lr_img[:, x_offset:x_offset + patch_size, 
                             y_offset:y_offset + patch_size]
            hr_patch = hr_img[:, 2*x_offset:2*x_offset + 2*patch_size,
                             2*y_offset:2*y_offset + 2*patch_size]
            
            low_res_patches.append(lr_patch)
            high_res_patches.append(hr_patch)
    
    # Stack and shuffle
    low_res_patches = torch.stack(low_res_patches)
    high_res_patches = torch.stack(high_res_patches)
    
    num_samples = low_res_patches.shape[0]
    shuffle_indices = torch.randperm(num_samples)
    
    return low_res_patches[shuffle_indices], high_res_patches[shuffle_indices]


def get_urban100_dataset(split='train', num_patches=10, patch_size=64):
    """
    Load Urban100 dataset for super-resolution (2x upscaling).
    
    Args:
        split: 'train' or 'test'
        num_patches: Number of patches to extract per image
        patch_size: Size of patches to extract (default: 64)
        
    Returns:
        Urban100Dataset with low-res and high-res image pairs
        
    Note:
        Expects the following files in MAIN_DATA_DIR/Urban100/:
        - low_res_train.pt, high_res_train.pt (for training)
        - low_res_test.pt, high_res_test.pt (for testing)
    """
    base_dir = os.path.join(MAIN_DATA_DIR, 'Urban100')
    
    if split == 'train':
        low_res_path = os.path.join(base_dir, 'low_res_train.pt')
        high_res_path = os.path.join(base_dir, 'high_res_train.pt')
    elif split == 'test':
        low_res_path = os.path.join(base_dir, 'low_res_test.pt')
        high_res_path = os.path.join(base_dir, 'high_res_test.pt')
    else:
        raise ValueError(f'Invalid split value: {split}. Must be "train" or "test".')
    
    # Generate patches
    lr_patches, hr_patches = generate_urban100_patches(
        low_res_path, high_res_path, 
        num_patches=num_patches,
        patch_size=patch_size
    )
    
    dataset = Urban100Dataset(lr_patches, hr_patches)
    return dataset
