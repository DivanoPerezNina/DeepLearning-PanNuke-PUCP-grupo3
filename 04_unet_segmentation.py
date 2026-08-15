"""
===============================================================================
U-NET ARCHITECTURE FROM SCRATCH & BIOIMAGE SEGMENTATION DATASET (PyTorch)
===============================================================================

1. OVERVIEW & THEORETICAL FOUNDATION OF U-NET
-------------------------------------------------------------------------------
The U-Net architecture was introduced by Olaf Ronneberger, Philipp Fischer, and
Thomas Brox in 2015 ("U-Net: Convolutional Networks for Biomedical Image Segmentation").
It was specifically engineered to address key challenges in biomedical image analysis:
  - Very limited availability of annotated training datasets (e.g., expert-labeled histology).
  - The requirement for precise pixel-level localization (semantic segmentation), not just image classification.
  - Touching or overlapping structures (e.g., adjacent cell membranes or histology glands).

2. STRUCTURAL FLOW OF THE U-NET ARCHITECTURE
-------------------------------------------------------------------------------
The U-Net gets its name from its symmetric "U"-shaped network pipeline, consisting of:

A. CONTRACTING PATH (Encoder / Downsampling Branch):
   - Purpose: Extracts high-level semantic context by gradually increasing the receptive field.
   - Components: Repeated blocks of two 3x3 Convolutions (followed by BatchNorm + ReLU activation),
     interspersed with 2x2 Max Pooling operations with stride 2.
   - Spatial Evolution: At each downsampling level, spatial dimensions (Height x Width) are halved,
     while the feature channel capacity doubles (e.g., 3 -> 64 -> 128 -> 256 -> 512 -> 1024).

B. BOTTLENECK (Bridge):
   - Purpose: Connects the deepest level of the encoder to the decoder.
   - Characteristics: Operates at the lowest spatial resolution (e.g., H/16 x W/16) with the highest
     feature channel depth (e.g., 1024 channels), compressing global contextual information.

C. EXPANDING PATH (Decoder / Upsampling Branch):
   - Purpose: Restores spatial resolution to match the original input image size while assembling
     a dense pixel-wise segmentation map.
   - Components: Repeated blocks of 2x2 Transposed Convolutions (or Bilinear Upsampling),
     Concatenation with encoder skip connections, and two 3x3 Convolutions (BatchNorm + ReLU).
   - Spatial Evolution: At each upsampling level, spatial dimensions double, while feature channel
     counts are halved (e.g., 1024 -> 512 -> 256 -> 128 -> 64).

D. SKIP CONNECTIONS (Bridge across Encoder & Decoder):
   - Purpose: Max pooling in the encoder discards fine-grained spatial position information in favor
     of translation invariance. Skip connections copy high-resolution spatial feature maps directly
     from the encoder and concatenate them with the upsampled features in the decoder.
   - Impact: Enables the network to preserve precise object boundaries (e.g., gland walls, cell membranes).

E. FINAL OUTPUT CONVOLUTION LAYER:
   - Purpose: Uses a 1x1 Convolution to map the final 64-channel feature map down to the target number
     of output classes (e.g., 1 channel for binary segmentation logits or K channels for multi-class).

===============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS (Standard PyTorch, NumPy, Napari, and Matplotlib libraries)
# -----------------------------------------------------------------------------
import os                                            # Operating system interface for path manipulation
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'          # Prevent OpenMP duplicate runtime initialization error on Windows

import random                                        # Random number generation for reproducible splits
import numpy as np                                   # Numerical array operations
import napari                                        # Interactive bioimage visualization GUI framework
import matplotlib.pyplot as plt                      # Static plot export engine
import torch                                         # PyTorch core tensor library
import torch.nn as nn                                # Neural network modules (Conv2d, MaxPool2d, etc.)
import torch.nn.functional as F                      # Functional activations and loss functions
from torch.utils.data import Dataset, DataLoader     # Abstract Dataset class and DataLoader batch manager



# =============================================================================
# PART 1: CUSTOM PYTORCH DATASET FOR BIOIMAGE SEGMENTATION (GLaS / PathMNIST)
# =============================================================================

# =============================================================================
# PART 1: CUSTOM PYTORCH DATASET FOR REAL BIOIMAGE PATHOLOGY (PathMNIST / GLaS)
# =============================================================================

class PathMNISTDataset(Dataset):
    """
    Custom PyTorch Dataset for Bioimage Pathology Segmentation (PathMNIST / MedMNIST & GLaS).
    
    Loads REAL biomedical pathology tissue patches (H&E stained colon histopathology)
    and derives real biological segmentation masks (0 = Stroma/Background, 1 = Nuclei/Epithelium).
    """

    def __init__(self, data_dir: str = None, num_samples: int = 100, image_size: int = 128, transform: bool = True):
        """
        Initializes the real bioimage dataset.
        
        Parameters:
            data_dir (str, optional): Custom folder directory containing 'images/' and 'masks/'.
                                      If None, automatically loads the REAL PathMNIST dataset.
            num_samples (int): Number of real samples to load from PathMNIST.
            image_size (int): Target spatial resolution (Height = Width = image_size).
            transform (bool): Whether to apply data augmentation (random flips).
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.data_dir = data_dir

        if data_dir is not None and os.path.exists(data_dir):
            self.images, self.masks = self._load_from_directory(data_dir)
        else:
            self.images, self.masks = self._load_real_pathmnist_data()

    def _load_real_pathmnist_data(self):
        """Loads REAL pathology tissue patches from the downloaded PathMNIST dataset."""
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache"))
        os.makedirs(cache_dir, exist_ok=True)
        npz_path = os.path.join(cache_dir, "pathmnist.npz")

        # Download REAL PathMNIST dataset from Zenodo if not present in local cache
        if not os.path.exists(npz_path):
            url = "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
            print(f"[PathMNISTDataset] Downloading REAL PathMNIST dataset from Zenodo to '{npz_path}'...")
            import urllib.request
            urllib.request.urlretrieve(url, npz_path)
            print("[PathMNISTDataset] Download complete!")

        # Load REAL pathology tissue patches
        npz_data = np.load(npz_path)
        raw_images = npz_data["train_images"][:self.num_samples] # Shape: (num_samples, 28, 28, 3)

        images_list = []
        masks_list = []

        from skimage.filters import threshold_otsu

        for i in range(len(raw_images)):
            # Normalize real H&E RGB image patch to [0, 1]
            img = raw_images[i].astype(np.float32) / 255.0

            # Derive real biological nucleus/epithelial segmentation target mask via stain thresholding
            hema_channel = img[:, :, 2] - img[:, :, 0]
            thresh_val = threshold_otsu(hema_channel) if hema_channel.max() > hema_channel.min() else 0.0
            mask = (hema_channel > thresh_val).astype(np.float32)

            images_list.append(img)
            masks_list.append(mask)

        return images_list, masks_list

    def __len__(self) -> int:
        """Returns the total number of real image-mask pairs in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Retrieves image and target mask at index `idx`.
        Converts NumPy arrays to PyTorch Tensors with shape [Channels, Height, Width]
        and resizes spatially to self.image_size.
        """
        img_np = self.images[idx]                    # Retrieve RGB image array (H, W, 3)
        mask_np = self.masks[idx]                    # Retrieve grayscale mask array (H, W)

        # Convert NumPy HWC -> PyTorch CHW tensor
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0) # [1, 3, H, W]
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()   # [1, 1, H, W]

        # Resize spatially to target U-Net input dimensions (e.g. 128x128)
        img_tensor = F.interpolate(img_tensor, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        mask_tensor = F.interpolate(mask_tensor, size=(self.image_size, self.image_size), mode='nearest').squeeze(0)

        # Apply spatial data augmentation (Synchronous random horizontal & vertical flips)
        if self.transform:
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])   # Flip width dimension
                mask_tensor = torch.flip(mask_tensor, dims=[2]) # Synchronously flip mask width
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])   # Flip height dimension
                mask_tensor = torch.flip(mask_tensor, dims=[1]) # Synchronously flip mask height

        return img_tensor, mask_tensor


# =============================================================================
# PART 2: U-NET BUILDING BLOCKS (DoubleConv, DownBlock, UpBlock)
# =============================================================================

class DoubleConv(nn.Module):
    """
    Standard U-Net Double Convolutional Block:
    [Conv2d(3x3) -> BatchNorm2d -> ReLU] -> [Conv2d(3x3) -> BatchNorm2d -> ReLU]
    Padded convolutions (padding=1) are used to maintain equal spatial dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()           # Initialize parent nn.Module
        self.double_conv = nn.Sequential(
            # First Convolution: Maps in_channels -> out_channels
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),            # Normalize activations to stabilize training
            nn.ReLU(inplace=True),                   # Non-linear activation function

            # Second Convolution: Maps out_channels -> out_channels
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),            # Normalize activations
            nn.ReLU(inplace=True)                    # Non-linear activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes input tensor through double 3x3 convolutions."""
        return self.double_conv(x)                   # Forward pass through Sequential block


class DownBlock(nn.Module):
    """
    U-Net Downsampling Block (Encoder Stage):
    MaxPool2d(2x2, stride=2) -> DoubleConv(in_channels, out_channels)
    Halves spatial dimensions (H, W) and doubles channel depth.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(DownBlock, self).__init__()             # Initialize parent nn.Module
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),   # Spatial downsampling (H/2, W/2)
            DoubleConv(in_channels, out_channels)    # Feature extraction double conv
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies 2x2 max pooling followed by double convolution."""
        return self.maxpool_conv(x)                  # Forward pass through downsampling block


class UpBlock(nn.Module):
    """
    U-Net Upsampling Block (Decoder Stage):
    ConvTranspose2d(2x2, stride=2) -> Concatenate Skip Connection -> DoubleConv
    Doubles spatial dimensions (H, W) and reduces channel depth.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(UpBlock, self).__init__()               # Initialize parent nn.Module
        # Transposed convolution upsamples spatial dimensions by a factor of 2
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Double convolution processes concatenated feature maps (upsampled + skip connection)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for UpBlock:
            x1: Feature tensor from lower layer (to be upsampled).
            x2: Skip connection feature tensor from corresponding encoder layer.
        """
        x1 = self.up(x1)                             # Upsample x1 spatially (H*2, W*2)

        # Pad x1 if spatial dimensions differ slightly due to odd input shapes
        diff_y = x2.size()[2] - x1.size()[2]         # Difference in height (H)
        diff_x = x2.size()[3] - x1.size()[3]         # Difference in width (W)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Concatenate skip connection (x2) and upsampled features (x1) along channel dimension (dim=1)
        x = torch.cat([x2, x1], dim=1)               # Channels: (in_channels // 2) + (in_channels // 2) = in_channels
        return self.conv(x)                          # Pass concatenated tensor through DoubleConv


# =============================================================================
# PART 3: COMPLETE U-NET MODULE WITH DETAILED TENSOR SHAPE COMMENTS
# =============================================================================

class UNet(nn.Module):
    """
    Full U-Net Architecture for Bioimage Segmentation.
    
    Includes 4 Downsampling levels (Encoder), 1 Bottleneck bridge,
    4 Upsampling levels (Decoder with Skip Connections), and 1 Output Conv.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        """
        Constructs all network layers for the U-Net architecture.

        Parameters:
            in_channels (int): Input image channels (3 for RGB, 1 for Grayscale).
            out_channels (int): Output segmentation map channels (1 for Binary logits).
        """
        super(UNet, self).__init__()                 # Initialize parent nn.Module
        self.in_channels = in_channels               # Store input channel count
        self.out_channels = out_channels             # Store output channel count

        # ---------------------------------------------------------------------
        # ENCODER (Contracting Path)
        # ---------------------------------------------------------------------
        self.inc = DoubleConv(in_channels, 64)       # Level 1: Initial Double Conv (3 -> 64)
        self.down1 = DownBlock(64, 128)              # Level 2: Downsample + Double Conv (64 -> 128)
        self.down2 = DownBlock(128, 256)             # Level 3: Downsample + Double Conv (128 -> 256)
        self.down3 = DownBlock(256, 512)             # Level 4: Downsample + Double Conv (256 -> 512)

        # ---------------------------------------------------------------------
        # BOTTLENECK (Bridge)
        # ---------------------------------------------------------------------
        self.down4 = DownBlock(512, 1024)            # Deepest Layer: Downsample + Double Conv (512 -> 1024)

        # ---------------------------------------------------------------------
        # DECODER (Expanding Path with Skip Connections)
        # ---------------------------------------------------------------------
        self.up1 = UpBlock(1024, 512)                # Up-level 1: Upsample + Concat + Double Conv (1024 -> 512)
        self.up2 = UpBlock(512, 256)                 # Up-level 2: Upsample + Concat + Double Conv (512 -> 256)
        self.up3 = UpBlock(256, 128)                 # Up-level 3: Upsample + Concat + Double Conv (256 -> 128)
        self.up4 = UpBlock(128, 64)                  # Up-level 4: Upsample + Concat + Double Conv (128 -> 64)

        # ---------------------------------------------------------------------
        # FINAL OUTPUT LAYER
        # ---------------------------------------------------------------------
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1) # 1x1 Conv mapping 64 channels -> target logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the U-Net architecture.
        
        Detailed Tensor Shape Transformations for Input [B, 3, 256, 256]:
            B = Batch Size
            C = Channel Depth
            H = Spatial Height
            W = Spatial Width
        """
        # =====================================================================
        # 1. ENCODER FORWARD PASS (Downsampling & Feature Extraction)
        # =====================================================================
        # INPUT TENSOR SHAPE:
        # [B, in_channels, H, W] -> Example: [B, 3, 256, 256]

        x1 = self.inc(x)
        # SHAPE AFTER INITIAL DOUBLE CONV (x1):
        # [B, 64, 256, 256]  <-- Saved as Skip Connection 1 (Encoder Level 1)

        x2 = self.down1(x1)
        # SHAPE AFTER DOWNBLOCK 1 (x2):
        # MaxPool(2x2) -> [B, 64, 128, 128] -> DoubleConv(64, 128) -> [B, 128, 128, 128]
        # <-- Saved as Skip Connection 2 (Encoder Level 2)

        x3 = self.down2(x2)
        # SHAPE AFTER DOWNBLOCK 2 (x3):
        # MaxPool(2x2) -> [B, 128, 64, 64] -> DoubleConv(128, 256) -> [B, 256, 64, 64]
        # <-- Saved as Skip Connection 3 (Encoder Level 3)

        x4 = self.down3(x3)
        # SHAPE AFTER DOWNBLOCK 3 (x4):
        # MaxPool(2x2) -> [B, 256, 32, 32] -> DoubleConv(256, 512) -> [B, 512, 32, 32]
        # <-- Saved as Skip Connection 4 (Encoder Level 4)

        # =====================================================================
        # 2. BOTTLENECK FORWARD PASS (Deepest Feature Representation)
        # =====================================================================
        x5 = self.down4(x4)
        # SHAPE AT BOTTLENECK (x5):
        # MaxPool(2x2) -> [B, 512, 16, 16] -> DoubleConv(512, 1024) -> [B, 1024, 16, 16]
        # Lowest spatial resolution (16x16), deepest semantic channels (1024)

        # =====================================================================
        # 3. DECODER FORWARD PASS (Upsampling & Skip Concatenations)
        # =====================================================================
        x = self.up1(x5, x4)
        # SHAPE AFTER UPBLOCK 1:
        # TransposeConv(x5 [B, 1024, 16, 16]) -> Upsampled [B, 512, 32, 32]
        # Concat with Skip Connection x4 [B, 512, 32, 32] along channels -> [B, 1024, 32, 32]
        # DoubleConv(1024, 512) -> [B, 512, 32, 32]

        x = self.up2(x, x3)
        # SHAPE AFTER UPBLOCK 2:
        # TransposeConv(x [B, 512, 32, 32]) -> Upsampled [B, 256, 64, 64]
        # Concat with Skip Connection x3 [B, 256, 64, 64] along channels -> [B, 512, 64, 64]
        # DoubleConv(512, 256) -> [B, 256, 64, 64]

        x = self.up3(x, x2)
        # SHAPE AFTER UPBLOCK 3:
        # TransposeConv(x [B, 256, 64, 64]) -> Upsampled [B, 128, 128, 128]
        # Concat with Skip Connection x2 [B, 128, 128, 128] along channels -> [B, 256, 128, 128]
        # DoubleConv(256, 128) -> [B, 128, 128, 128]

        x = self.up4(x, x1)
        # SHAPE AFTER UPBLOCK 4:
        # TransposeConv(x [B, 128, 128, 128]) -> Upsampled [B, 64, 256, 256]
        # Concat with Skip Connection x1 [B, 64, 256, 256] along channels -> [B, 128, 256, 256]
        # DoubleConv(128, 64) -> [B, 64, 256, 256]

        # =====================================================================
        # 4. FINAL OUTPUT CONVOLUTION
        # =====================================================================
        logits = self.outc(x)
        # SHAPE AFTER FINAL 1x1 CONVOLUTION (logits):
        # [B, out_channels, 256, 256] -> Example: [B, 1, 256, 256] for binary segmentation

        return logits


# =============================================================================
# PART 4: LOSS FUNCTION & TRAINING DEMONSTRATION
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary semantic segmentation.
    Measures spatial overlap between predicted probability map and ground-truth mask.
    Dice = (2 * |P ∩ G|) / (|P| + |G|)
    """

    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()              # Initialize parent nn.Module
        self.smooth = smooth                         # Smoothing factor to prevent division by zero

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes Dice loss from raw network logits and binary targets."""
        probs = torch.sigmoid(logits)                # Convert unnormalized logits to probabilities [0, 1]
        
        # Flatten predictions and targets to 1D vectors per batch item
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Compute intersection and union sums
        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)

        return 1.0 - dice_coeff                      # Return Dice Loss (1.0 - Dice Coefficient)


def train_unet_demo():
    """
    Demonstrates model instantiation, forward pass tensor shape verification,
    and a short training loop on the custom synthetic GLaS BioImage dataset.
    """
    print("==========================================================")
    print("STARTING U-NET BIOIMAGE SEGMENTATION TRAINING DEMO")
    print("==========================================================")

    # 1. Hardware device selection (GPU CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Execution Device: {device}")

    # 2. Instantiate Dataset and DataLoaders
    print("\n[Step 1] Loading REAL PathMNIST (MedMNIST) pathology tissue dataset...")
    train_dataset = PathMNISTDataset(num_samples=32, image_size=128, transform=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    print(f" -> Dataset size: {len(train_dataset)} real samples, Batch size: 4")

    # 3. Instantiate U-Net Architecture
    print("\n[Step 2] Building U-Net Architecture from scratch...")
    model = UNet(in_channels=3, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Total Trainable Parameters: {total_params:,}")

    # 4. Verify Forward Pass Tensor Shapes
    print("\n[Step 3] Verifying Forward Pass Tensor Transformations...")
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f" -> Input Tensor Shape : {list(dummy_input.shape)}")
    print(f" -> Output Logits Shape: {list(dummy_output.shape)}")
    assert dummy_output.shape == (2, 1, 128, 128), "Output shape mismatch!"
    print(" -> Tensor shape verification SUCCESSFUL!")

    # 5. Define Loss Function & Optimizer
    criterion_bce = nn.BCEWithLogitsLoss()           # Standard Binary Cross Entropy with Logits
    criterion_dice = DiceLoss()                     # Soft Dice Loss for region overlap
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Adam Optimizer

    # 6. Short Demonstration Training Loop (3 Epochs)
    print("\n[Step 4] Executing 3-Epoch Training Loop...")
    epochs = 3
    model.train()                                    # Set model to training mode

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()                    # Reset gradients from previous step
            logits = model(images)                   # Forward pass through U-Net

            # Calculate combined Loss (BCE + Dice Loss)
            loss_bce = criterion_bce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = loss_bce + loss_dice             # Total loss

            loss.backward()                          # Backpropagate loss gradients
            optimizer.step()                         # Update model parameters

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f" -> Epoch [{epoch}/{epochs}] Complete | Average Batch Loss: {epoch_loss:.4f}")

    # 7. Qualitative Visual Inspection with Napari Interactive Viewer
    print("\n[Step 5] Rendering Interactive Napari Bioimage Visualization...")
    model.eval()                                     # Set model to evaluation mode
    sample_img, sample_mask = train_dataset[0]
    with torch.no_grad():
        input_tensor = sample_img.unsqueeze(0).to(device) # Add batch dimension [1, 3, 256, 256]
        pred_logit = model(input_tensor)
        pred_prob = torch.sigmoid(pred_logit).squeeze().cpu().numpy() # Convert logit -> probability map [0, 1]

    # Convert CHW image tensor to HWC NumPy format for Napari
    display_img = sample_img.permute(1, 2, 0).numpy()
    display_mask = sample_mask.squeeze().numpy().astype(np.uint32)
    binary_pred = (pred_prob > 0.5).astype(np.uint32)

    # Export a static PNG summary figure for documentation
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(display_img); axes[0].set_title("1. Input H&E Bioimage"); axes[0].axis('off')
    axes[1].imshow(display_mask, cmap='gray'); axes[1].set_title("2. Ground-Truth Mask"); axes[1].axis('off')
    im2 = axes[2].imshow(pred_prob, cmap='magma'); axes[2].set_title("3. U-Net Prediction Probability"); axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    output_png = "unet_segmentation_output.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> Static summary figure saved to '{output_png}'.")

    # Initialize Napari interactive viewer
    print(" -> Launching Napari Interactive Viewer...")
    viewer = napari.Viewer(title="PyTorch U-Net Bioimage Segmentation - Napari Viewer")

    # Layer 1: Input H&E RGB Bioimage
    viewer.add_image(
        display_img,
        name="1. Input H&E Bioimage",
        rgb=True
    )

    # Layer 2: Ground-Truth Gland Mask (Labels Layer)
    viewer.add_labels(
        display_mask,
        name="2. Ground-Truth Gland Mask",
        opacity=0.5
    )

    # Layer 3: U-Net Probability Map (Continuous heatmap)
    viewer.add_image(
        pred_prob,
        name="3. U-Net Continuous Probability Map",
        colormap="magma",
        opacity=0.7,
        visible=False
    )

    # Layer 4: U-Net Binary Segmentation Mask (> 0.5 threshold)
    viewer.add_labels(
        binary_pred,
        name="4. U-Net Binary Prediction (Threshold > 0.5)",
        opacity=0.5
    )

    print(" -> Napari viewer active. Close the Napari window to finish execution.")

    # Run Napari GUI loop if not running headless
    if plt.get_backend().lower() != 'agg':
        napari.run()
    else:
        print(" -> Non-interactive mode ('Agg'); skipped Napari GUI event loop.")

    print("\nU-Net segmentation pipeline execution complete!")


# Standard Python main entry point guard
if __name__ == "__main__":
    train_unet_demo()
