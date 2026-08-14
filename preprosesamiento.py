"""
===============================================================================
PREPROCESAMIENTO DEL DATASET PANNUKE (SEGMENTACION BINARIA)
===============================================================================
Este script asume que ya descargaste manualmente un mirror de PanNuke desde
Kaggle y lo dejaste en `data/` con esta estructura:

    data/train/images/*.png      data/train/masks/*.png
    data/validate/images/*.png   data/validate/masks/*.png

DIFERENCIA CON EL PANNUKE OFICIAL:
  El PanNuke oficial (TIA Centre / Warwick, o el mirror de Kaggle
  theredlad/pannuke-dataset-experimental-data) distribuye 3 folds en .npy
  con mascaras de 6 canales (5 tipos de nucleo + fondo), preservando las
  clases Neoplasico/Inflamatorio/Conectivo/Muerto/Epitelial.

  El mirror que se uso aqui ya viene como PNG con mascaras BINARIAS
  (0 = fondo, 255 = nucleo, sin distincion de tipo) y solo 2 splits
  (train/validate, sin test). Por eso este script:
    1. Genera el split de test faltante partiendo `validate` 50/50.
    2. Trabaja con segmentacion BINARIA (1 clase "nucleo" + fondo), no con
       las 6 clases del PanNuke completo.

  Si en el futuro se consigue el .npy oficial multiclase, usar
  `descargar_pannuke.py` + un script de estructuracion multiclase en su
  lugar (ver historial de este archivo).
===============================================================================
"""

import os
import glob
import random
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# =============================================================================
# CONSTANTES
# =============================================================================

DATA_DIR = "data"
SPLIT_DIRS = {"train": "train", "val": "validate", "test": "test"}
TEST_SPLIT_SEED = 42
TEST_SPLIT_RATIO = 0.5  # proporcion de "validate" que se mueve a "test"


# =============================================================================
# SECCION 1: GENERAR EL SPLIT DE TEST (FALTANTE) A PARTIR DE VALIDATE
# =============================================================================

def split_validate_into_val_test(
    data_dir: str = DATA_DIR,
    test_ratio: float = TEST_SPLIT_RATIO,
    seed: int = TEST_SPLIT_SEED,
) -> None:
    """
    El mirror descargado solo trae train/ y validate/ (sin test/). Aqui se
    separa la mitad de validate/ (elegida al azar con semilla fija, para que
    el split sea reproducible) y se MUEVE a test/, dejando la otra mitad en
    validate/. Es idempotente: si test/ ya existe y tiene archivos, no hace
    nada.
    """
    val_images_dir = os.path.join(data_dir, "validate", "images")
    val_masks_dir = os.path.join(data_dir, "validate", "masks")
    test_images_dir = os.path.join(data_dir, "test", "images")
    test_masks_dir = os.path.join(data_dir, "test", "masks")

    if os.path.isdir(test_images_dir) and len(os.listdir(test_images_dir)) > 0:
        print("[OK] 'data/test' ya existe, se omite el split de validate.")
        return

    filenames = sorted(os.path.basename(p) for p in glob.glob(os.path.join(val_images_dir, "*.png")))
    if not filenames:
        raise RuntimeError(f"No se encontraron imagenes en '{val_images_dir}'.")

    rng = random.Random(seed)
    rng.shuffle(filenames)
    n_test = int(len(filenames) * test_ratio)
    test_filenames = set(filenames[:n_test])

    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_masks_dir, exist_ok=True)

    print(f"[INFO] Moviendo {len(test_filenames)}/{len(filenames)} pares de 'validate' a 'test'...")
    for fname in test_filenames:
        os.replace(os.path.join(val_images_dir, fname), os.path.join(test_images_dir, fname))
        os.replace(os.path.join(val_masks_dir, fname), os.path.join(test_masks_dir, fname))

    print(f"[OK] 'test' listo con {len(test_filenames)} pares; 'validate' quedo con {len(filenames) - len(test_filenames)}.")


# =============================================================================
# SECCION 2: VALIDACION + REPORTE DE BALANCE NUCLEO/FONDO
# =============================================================================

def validate_and_report_splits(data_dir: str = DATA_DIR) -> Dict[str, Dict]:
    """
    Verifica emparejamiento imagen-mascara por split y reporta el porcentaje
    de pixeles de nucleo vs fondo en train, para detectar el desbalance de
    clases tipico en segmentacion de nucleos (el fondo domina la imagen).
    """
    print("\n" + "=" * 60)
    print(" REPORTE DE VALIDACION (PanNuke binario)")
    print("=" * 60)

    report: Dict[str, Dict] = {}
    nucleus_px = 0
    total_px = 0

    for split_key, folder in SPLIT_DIRS.items():
        images_dir = os.path.join(data_dir, folder, "images")
        masks_dir = os.path.join(data_dir, folder, "masks")
        img_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        matched = len(img_paths) == len(mask_paths) and len(img_paths) > 0
        report[split_key] = {"num_samples": len(img_paths), "matched": matched}
        print(f"  - {split_key:5s}: {len(img_paths)} muestras | pares OK: {matched}")

        if split_key == "train":
            for mp in mask_paths:
                arr = np.array(Image.open(mp))
                nucleus_px += int((arr > 0).sum())
                total_px += arr.size

    pct_nucleus = 100.0 * nucleus_px / total_px if total_px > 0 else 0.0
    print("\n  Distribucion de pixeles (split train):")
    print(f"    Fondo : {100 - pct_nucleus:6.2f}%")
    print(f"    Nucleo: {pct_nucleus:6.2f}%")
    print("=" * 60 + "\n")

    report["pct_nucleus_train"] = pct_nucleus
    return report


# =============================================================================
# SECCION 3: DATASET Y DATALOADERS (SEGMENTACION BINARIA)
# =============================================================================

class PanNukeBinarySegmentationDataset(Dataset):
    """
    Dataset de segmentacion binaria: nucleo (1) vs fondo (0). La mascara se
    devuelve como FloatTensor (1, H, W) en [0, 1], el formato que espera
    `nn.BCEWithLogitsLoss` (a diferencia de una mascara multiclase de
    indices para `nn.CrossEntropyLoss`).
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        target_size: Tuple[int, int] = (256, 256),
        transform: bool = True,
    ):
        assert len(image_paths) == len(mask_paths), "El numero de imagenes y mascaras debe coincidir."
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.target_size = target_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_synchronous_transforms(self, image_np: np.ndarray, mask_np: np.ndarray):
        """Flips + rotaciones de 90 grados, sincronizadas entre imagen y mascara."""
        if random.random() > 0.5:
            image_np = np.fliplr(image_np)
            mask_np = np.fliplr(mask_np)
        if random.random() > 0.5:
            image_np = np.flipud(image_np)
            mask_np = np.flipud(mask_np)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            image_np = np.rot90(image_np, k, axes=(0, 1))
            mask_np = np.rot90(mask_np, k, axes=(0, 1))
        return image_np.copy(), mask_np.copy()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_pil = Image.open(self.image_paths[idx]).convert("RGB")
        mask_pil = Image.open(self.mask_paths[idx]).convert("L")

        img_pil = img_pil.resize(self.target_size, Image.BILINEAR)
        mask_pil = mask_pil.resize(self.target_size, Image.NEAREST)

        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        mask_np = (np.array(mask_pil) > 0).astype(np.float32)  # binarizar: 0/1

        if self.transform:
            img_np, mask_np = self._apply_synchronous_transforms(img_np, mask_np)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)      # (3, H, W) float
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)        # (1, H, W) float -> BCEWithLogitsLoss

        return img_tensor, mask_tensor


def create_pannuke_dataloaders(
    data_dir: str = DATA_DIR,
    batch_size: int = 8,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    datasets = {}
    for split_key, folder in SPLIT_DIRS.items():
        images_dir = os.path.join(data_dir, folder, "images")
        masks_dir = os.path.join(data_dir, folder, "masks")
        img_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        datasets[split_key] = PanNukeBinarySegmentationDataset(
            img_paths, mask_paths, transform=(split_key == "train")
        )

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=================================================================")
    print(" PREPROCESAMIENTO PANNUKE (BINARIO: NUCLEO VS FONDO) ")
    print("=================================================================")

    split_validate_into_val_test()
    validate_and_report_splits()

    print("[INFO] Construyendo DataLoaders (train/validate/test)...")
    train_loader, val_loader, test_loader = create_pannuke_dataloaders(batch_size=8)

    images, masks = next(iter(train_loader))
    print(f"\n[OK] Batch de entrenamiento cargado:")
    print(f"     Imagen : {tuple(images.shape)}, dtype={images.dtype}")
    print(f"     Mascara: {tuple(masks.shape)}, dtype={masks.dtype}, valores unicos: {torch.unique(masks).tolist()}")

    print("\n[OK] Preprocesamiento completado.")
    print("     Para entrenar: UNet(in_channels=3, out_channels=1) + nn.BCEWithLogitsLoss()")
    print("     (segmentacion binaria: sin distincion de tipo de nucleo).")
