"""
===============================================================================
BIOMEDICAL DATASET EXPLORATION & PYTORCH DATALOADERS
===============================================================================

This script provides a complete and standalone workflow for downloading, 
structuring, augmenting, and batch loading biomedical datasets in PyTorch 
with interactive visualization in Napari.

Script Sections:
  1. Automated download and inspection of a biomedical nuclei dataset.
  2. Implementation of a custom `torch.utils.data.Dataset` class.
  3. Reproducible split (Train / Val / Test) and `DataLoader` generation.
  4. Interactive multi-channel inspection in Napari (with headless fallback).

===============================================================================
"""

import os
import sys
import glob
import random
import urllib.request
import zipfile
from typing import Tuple, List, Optional

# Force UTF-8 encoding on Windows consoles
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Avoid duplicate OpenMP initialization errors on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# =============================================================================
# SECTION 1: DATA DOWNLOAD AND STRUCTURING (Data Science Bowl / Synthetic)
# =============================================================================

def generate_synthetic_nuclei_dataset(output_dir: str, num_samples: int = 50, image_size: int = 256):
    """
    Generates a realistic synthetic fluorescence microscopy dataset and binary masks.
    Ensures script execution without requiring an internet connection.
    """
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    print(f"  [Synthetic] Generating {num_samples} synthetic biomedical samples in '{output_dir}'...")

    for i in range(num_samples):
        # Create blank canvas
        img = np.zeros((image_size, image_size, 3), dtype=np.float32)
        mask = np.zeros((image_size, image_size), dtype=np.uint8)

        # Generate between 15 and 35 nuclei per image
        num_nuclei = random.randint(15, 35)
        xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size))

        for _ in range(num_nuclei):
            cx = random.randint(20, image_size - 20)
            cy = random.randint(20, image_size - 20)
            rx = random.randint(8, 18)
            ry = random.randint(8, 18)
            angle = random.uniform(0, np.pi)

            # Elliptic equation
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = cos_a * (xx - cx) + sin_a * (yy - cy)
            y_rot = -sin_a * (xx - cx) + cos_a * (yy - cy)
            dist = (x_rot ** 2) / (rx ** 2) + (y_rot ** 2) / (ry ** 2)

            ellipse_mask = dist <= 1.0
            mask[ellipse_mask] = 1

            # Add intensity to image (blue/DAPI channel and green/membrane channel)
            intensity = random.uniform(0.6, 1.0)
            img[..., 0] += (ellipse_mask * intensity * 0.2).astype(np.float32)  # Residual red
            img[..., 1] += (ellipse_mask * intensity * 0.4).astype(np.float32)  # Green
            img[..., 2] += (ellipse_mask * intensity * 0.9).astype(np.float32)  # Intense blue

        # Gaussian background noise and texture
        noise = np.random.normal(0.05, 0.02, img.shape)
        img = np.clip(img + noise, 0.0, 1.0)
        img_uint8 = (img * 255).astype(np.uint8)

        # Save to disk
        img_name = f"sample_{i+1:03d}.png"
        mask_name = f"sample_{i+1:03d}.png"

        Image.fromarray(img_uint8).save(os.path.join(images_dir, img_name))
        Image.fromarray(mask * 255).save(os.path.join(masks_dir, mask_name))


def resolve_data_dir(data_dir: str = "data/raw/nuclei") -> str:
    """Checks if a formatted dataset exists (from Script 06) and prefers it over synthetic data."""
    formatted_alt = "data/raw/nuclei_formatted"
    if data_dir == "data/raw/nuclei" and os.path.exists(os.path.join(formatted_alt, "images")):
        num_fmt = len(glob.glob(os.path.join(formatted_alt, "images", "*.png")))
        if num_fmt > 0:
            return formatted_alt
    return data_dir


def download_and_extract_nuclei_dataset(data_dir: str = "data/raw/nuclei") -> Tuple[str, str]:
    """
    Downloads or locally generates the cell nuclei segmentation dataset.
    Organizes files into `data_dir/images` and `data_dir/masks`.
    """
    data_dir = resolve_data_dir(data_dir)

    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    if os.path.exists(images_dir) and os.path.exists(masks_dir):
        num_imgs = len(glob.glob(os.path.join(images_dir, "*.png")))
        if num_imgs > 0:
            print(f"[OK] Existing dataset detected in '{data_dir}' with {num_imgs} images.")
            return images_dir, masks_dir

    print(f"[INFO] Generating biomedical nuclei dataset in '{data_dir}'...")
    generate_synthetic_nuclei_dataset(data_dir, num_samples=60, image_size=256)
    print("[OK] Dataset successfully prepared.")

    return images_dir, masks_dir


# =============================================================================
# SECTION 2: DEFINITION OF CUSTOM PYTORCH DATASET
# =============================================================================

class BioImageSegmentationDataset(Dataset):
    """
    Custom `torch.utils.data.Dataset` class for biomedical semantic segmentation.
    
    Processes pairs of images (RGB or Grayscale) and annotation masks (Ground Truth),
    applying scaling, normalization, and synchronous data augmentation transformations.
    """

    def __init__(
        self, 
        image_paths: List[str], 
        mask_paths: List[str], 
        target_size: Tuple[int, int] = (256, 256),
        transform: bool = True
    ):
        """
        Parameters:
            image_paths (List[str]): Paths to input image files.
            mask_paths (List[str]): Paths to ground truth mask files.
            target_size (Tuple[int, int]): Output spatial dimension (Height, Width).
            transform (bool): Indicates whether random data augmentation will be applied.
        """
        assert len(image_paths) == len(mask_paths), "The number of images and masks must be identical."
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.target_size = target_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_synchronous_transforms(
        self, 
        image_np: np.ndarray, 
        mask_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies random spatial transformations symmetrically to the image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image_np = np.fliplr(image_np)
            mask_np = np.fliplr(mask_np)

        # Random vertical flip
        if random.random() > 0.5:
            image_np = np.flipud(image_np)
            mask_np = np.flipud(mask_np)

        # Right-angle rotations (90, 180, 270 deg)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            image_np = np.rot90(image_np, k, axes=(0, 1))
            mask_np = np.rot90(mask_np, k, axes=(0, 1))

        return image_np.copy(), mask_np.copy()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Load image and mask from disk using PIL
        img_pil = Image.open(self.image_paths[idx]).convert("RGB")
        mask_pil = Image.open(self.mask_paths[idx]).convert("L")

        # 2. Resize to target dimension (256x256)
        img_pil = img_pil.resize(self.target_size, Image.BILINEAR)
        mask_pil = mask_pil.resize(self.target_size, Image.NEAREST)

        # 3. Convert to NumPy arrays in range [0.0, 1.0]
        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        mask_np = (np.array(mask_pil, dtype=np.float32) > 0).astype(np.float32)

        # 4. Apply synchronous augmentation during training
        if self.transform:
            img_np, mask_np = self._apply_synchronous_transforms(img_np, mask_np)

        # 5. Rearrange NumPy axes (H, W, C) to PyTorch tensors (C, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # Shape: (3, H, W)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)     # Shape: (1, H, W)

        return img_tensor, mask_tensor


# =============================================================================
# SECTION 3: DATA SPLITTING AND DATALOADER GENERATION
# =============================================================================

def create_dataloaders(
    data_dir: str = "data/raw/nuclei", 
    batch_size: int = 8, 
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Performs reproducible stochastic splitting (80% Train, 10% Val, 10% Test)
    and instantiates PyTorch DataLoaders configured with `batch_size=8`.
    """
    data_dir = resolve_data_dir(data_dir)

    # Set seeds for complete reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

    total_samples = len(image_paths)
    assert total_samples > 0, "No images found in the specified directory."

    # Pair and shuffle order stochastically
    combined = list(zip(image_paths, mask_paths))
    random.shuffle(combined)
    shuffled_img_paths, shuffled_mask_paths = zip(*combined)

    # Calculate indices for 80% / 10% / 10% split
    train_end = int(0.80 * total_samples)
    val_end = int(0.90 * total_samples)

    train_imgs, train_masks = shuffled_img_paths[:train_end], shuffled_mask_paths[:train_end]
    val_imgs, val_masks = shuffled_img_paths[train_end:val_end], shuffled_mask_paths[train_end:val_end]
    test_imgs, test_masks = shuffled_img_paths[val_end:], shuffled_mask_paths[val_end:]

    print(f"\n📊 Dataset Split ({total_samples} total samples):")
    print(f"  • Training   (80%): {len(train_imgs)} images")
    print(f"  • Validation (10%): {len(val_imgs)} images")
    print(f"  • Testing    (10%): {len(test_imgs)} images")

    # Instantiate Dataset objects
    train_dataset = BioImageSegmentationDataset(train_imgs, train_masks, transform=True)
    val_dataset = BioImageSegmentationDataset(val_imgs, val_masks, transform=False)
    test_dataset = BioImageSegmentationDataset(test_imgs, test_masks, transform=False)

    # Instantiate PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =============================================================================
# SECTION 4: INTERACTIVE VISUALIZATION IN NAPARI (WITH HEADLESS FALLBACK)
# =============================================================================

def visualize_batch_in_napari(train_loader: DataLoader):
    """
    Interactively displays a training batch in Napari.
    Overlays the biomedical image layer and mask labels layer (opacity 0.5).
    """
    print("\n🔍 Inspecting the first training batch...")
    images, masks = next(iter(train_loader))

    print(f"  • Image Tensor Shape (Batch, C, H, W): {images.shape}")
    print(f"  • Mask Tensor Shape (Batch, 1, H, W): {masks.shape}")
    print(f"  • Image Value Range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"  • Unique Values in Masks: {torch.unique(masks).tolist()}")

    # Prepare NumPy arrays for Napari (Shape: Batch, H, W, C or Batch, H, W)
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)
    masks_np = masks.squeeze(1).cpu().numpy().astype(int) # (B, H, W) with integer labels

    # Defensive visualization in Napari (handling for GUI-less / Headless environments)
    try:
        import napari
        print("\n[+] Starting interactive Napari viewer...")
        viewer = napari.Viewer(title="Bioimage Analysis - DataLoader Batch Exploration")

        # Add image layer (fluorescence / H&E)
        viewer.add_image(
            images_np, 
            name="Biomedical Images (Batch)", 
            rgb=True
        )

        # Overlay label layer (Ground Truth) with opacity 0.5
        viewer.add_labels(
            masks_np, 
            name="Segmentation Masks (Ground Truth)", 
            opacity=0.5
        )

        print("[OK] Napari started successfully. Close the viewer window to finish the script.")
        napari.run()

    except Exception as err:
        print(f"[!] Headless environment or missing GUI support detected for Napari ({err}).")
        print("[INFO] Generating static fallback figure with Matplotlib ('dataset_exploration_output.png')...")

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for idx in range(min(4, images_np.shape[0])):
            # Display Image
            axes[0, idx].imshow(images_np[idx])
            axes[0, idx].set_title(f"Sample {idx+1}: RGB Image")
            axes[0, idx].axis("off")

            # Overlay Mask on Grayscale
            axes[1, idx].imshow(images_np[idx], cmap="gray")
            axes[1, idx].imshow(masks_np[idx], cmap="jet", alpha=0.5)
            axes[1, idx].set_title(f"Sample {idx+1}: Mask (Alpha=0.5)")
            axes[1, idx].axis("off")

        plt.suptitle("Biomedical DataLoader Inspection (Batch Size = 8)", fontsize=16)
        plt.tight_layout()
        output_plot = "dataset_exploration_output.png"
        plt.savefig(output_plot, dpi=150)
        plt.close()
        print(f"[OK] Static visualization saved to '{output_plot}'.")


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    print("=================================================================")
    print("  DLBA - DATASET EXPLORATION AND DATALOADERS IN PYTORCH ")
    print("=================================================================")

    # Step 1: Resolve dataset directory (prefers data/raw/nuclei_formatted if present)
    data_dir = resolve_data_dir("data/raw/nuclei")
    download_and_extract_nuclei_dataset(data_dir=data_dir)

    # Steps 2 & 3: Create splits and DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        seed=42
    )

    # Step 4: Visualize batch in Napari
    visualize_batch_in_napari(train_loader)

    print("\n[OK] Script 05 completed successfully.")


