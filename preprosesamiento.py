"""
===============================================================================
SCRIPT 07: PREPROCESAMIENTO DEL DATASET PANNUKE (SEGMENTACIÓN MULTICLASE)
===============================================================================
Extiende el pipeline de descarga/estructuración visto en 05 y 06 para PanNuke:
7,904 parches H&E de 256x256 px, 19 tipos de tejido, núcleos anotados en
5 clases (Neoplásico, Inflamatorio, Conectivo, Muerto, Epitelial) + fondo.

DIFERENCIA CLAVE CON 05/06:
  - 06 (DATASETS_REGISTRY) solo tiene registrado DSB2018 y funde todas las
    máscaras de instancia en una sola máscara BINARIA (0/255).
  - 05 (BioImageSegmentationDataset) fuerza `.convert("L")` + `> 0`, es decir,
    también binariza.
  - PanNuke necesita preservar las 5 clases, así que ambos se reescriben aquí
    en vez de reusarse tal cual.

DESCARGA:
  PanNuke no se puede automatizar con un solo pooch.retrieve() confiable:
  - La página oficial del TIA Centre (Univ. de Warwick) se reorganizó y ya
    no expone un enlace .zip directo por fold: https://warwick.ac.uk/TIA
  - El espejo de Kaggle requiere autenticación (API key / cuenta):
    https://www.kaggle.com/datasets/theredlad/pannuke-dataset-experimental-data
  Por eso este script asume que YA descargaste y descomprimiste los 3 folds
  manualmente en `raw_data_dir/foldN/{images.npy,masks.npy,types.npy}`.
  Si no encuentra esos archivos, genera datos SINTÉTICOS con la misma forma
  para que puedas probar el pipeline completo sin tener el dataset real.

FORMATO OFICIAL POR FOLD:
  images.npy -> (N, 256, 256, 3)  uint8   parches RGB H&E
  masks.npy  -> (N, 256, 256, 6)  canales 0-4 = ID de instancia por clase
                (0 = sin núcleo, >0 = ID de instancia); canal 5 = fondo
  types.npy  -> (N,) string        tipo de tejido de cada parche (no se usa
                en este script, pero queda disponible para análisis futuro)

SPLIT OFICIAL (a diferencia del split aleatorio 80/10/10 que usa 05):
  Fold 1 = Train | Fold 2 = Validation | Fold 3 = Test
===============================================================================
"""

import os
import sys
import glob
import random
from typing import Tuple, List, Dict, Any

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# =============================================================================
# CONSTANTES DEL DATASET
# =============================================================================

# Índice 0 = fondo. Índices 1-5 = canales 0-4 de masks.npy, en ese orden.
CLASS_NAMES = ["Fondo", "Neoplasico", "Inflamatorio", "Conectivo", "Muerto", "Epitelial"]
NUM_CLASSES = len(CLASS_NAMES)  # 6 (incluye fondo) -> UNet(out_channels=6)

FOLD_TO_SPLIT = {"fold1": "train", "fold2": "val", "fold3": "test"}


# =============================================================================
# SECCION 1: LOCALIZAR DATOS REALES O GENERAR SINTETICOS DE RESPALDO
# =============================================================================

def _generate_synthetic_pannuke_fold(num_samples: int, seed: int) -> Dict[str, np.ndarray]:
    """Genera un fold sintetico con la misma forma que el PanNuke real, para
    poder probar el pipeline de estructuracion/Dataset sin el dataset real."""
    rng = np.random.default_rng(seed)
    images = np.full((num_samples, 256, 256, 3), 235, dtype=np.uint8)  # fondo tipo "vidrio"
    masks = np.zeros((num_samples, 256, 256, 6), dtype=np.int32)
    types = rng.choice(["Colon", "Breast", "Lung", "Kidney", "Skin"], size=num_samples).astype(object)

    for i in range(num_samples):
        n_nuclei = rng.integers(20, 60)
        instance_id = 1
        for _ in range(n_nuclei):
            class_idx = rng.integers(0, 5)  # 0..4 -> Neoplasico..Epitelial
            cy, cx = rng.integers(15, 241, size=2)
            r = rng.integers(5, 14)
            yy, xx = np.ogrid[:256, :256]
            circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            masks[i, :, :, class_idx][circle] = instance_id
            instance_id += 1
            images[i][circle] = [60 + class_idx * 30, 30, 110 - class_idx * 10]  # tono distinto por clase

        no_nucleus = masks[i, :, :, :5].sum(axis=-1) == 0
        masks[i, :, :, 5][no_nucleus] = 1  # canal 5 = fondo

    return {"images": images, "masks": masks, "types": types}


def locate_or_generate_raw_folds(
    raw_data_dir: str = "data/raw/pannuke_source",
    num_synthetic_samples: int = 40,
) -> Dict[str, str]:
    """
    Busca foldN/{images.npy,masks.npy,types.npy} ya descargados en `raw_data_dir`.
    Si no existen, genera 3 folds sinteticos alli mismo para poder probar el
    resto del pipeline offline.

    Returns:
        Dict[str, str]: {"fold1": ruta, "fold2": ruta, "fold3": ruta}
    """
    fold_paths = {}
    all_present = True
    for fold_name in FOLD_TO_SPLIT:
        fold_dir = os.path.join(raw_data_dir, fold_name)
        required = [os.path.join(fold_dir, f) for f in ("images.npy", "masks.npy", "types.npy")]
        fold_paths[fold_name] = fold_dir
        if not all(os.path.exists(p) for p in required):
            all_present = False

    if all_present:
        print(f"[OK] Folds reales encontrados en '{raw_data_dir}'.")
        return fold_paths

    print(f"[!] No se encontraron los 3 folds reales en '{raw_data_dir}'.")
    print("[INFO] Descargalos manualmente desde el TIA Centre (https://warwick.ac.uk/TIA)")
    print("[INFO] o desde el espejo de Kaggle (requiere cuenta):")
    print("[INFO]   https://www.kaggle.com/datasets/theredlad/pannuke-dataset-experimental-data")
    print(f"[INFO] Generando {num_synthetic_samples} muestras SINTETICAS por fold para poder probar el pipeline...")

    for i, fold_name in enumerate(FOLD_TO_SPLIT):
        fold_dir = fold_paths[fold_name]
        os.makedirs(fold_dir, exist_ok=True)
        synthetic = _generate_synthetic_pannuke_fold(num_synthetic_samples, seed=100 + i)
        np.save(os.path.join(fold_dir, "images.npy"), synthetic["images"])
        np.save(os.path.join(fold_dir, "masks.npy"), synthetic["masks"])
        np.save(os.path.join(fold_dir, "types.npy"), synthetic["types"])
        print(f"  -> '{fold_name}' sintetico generado en '{fold_dir}'.")

    return fold_paths


# =============================================================================
# SECCION 2: ESTRUCTURACION - CONVERTIR .npy A images/ + masks/ (PNG INDEXADO)
# =============================================================================

def convert_pannuke_mask_to_indexed(mask_6ch: np.ndarray) -> np.ndarray:
    """
    Convierte una mascara PanNuke (256, 256, 6) en una mascara semantica
    indexada (256, 256) con valores 0-5 (0=Fondo, 1-5=clases).

    Los canales 0-4 de PanNuke codifican ID de instancia por clase (0 = sin
    nucleo); aqui solo nos importa "hay nucleo de esta clase o no" (semantica,
    no instancia), asi que cualquier valor > 0 se colapsa a la clase.
    """
    h, w = mask_6ch.shape[:2]
    indexed = np.zeros((h, w), dtype=np.uint8)
    for class_channel in range(5):
        indexed[mask_6ch[:, :, class_channel] > 0] = class_channel + 1
    return indexed


def structure_pannuke_folds(
    fold_paths: Dict[str, str],
    output_dir: str = "data/raw/pannuke_formatted",
) -> Dict[str, Tuple[str, str]]:
    """
    Convierte cada fold de .npy a PNGs organizados como:
        output_dir/<split>/images/*.png   (RGB)
        output_dir/<split>/masks/*.png    (escala de grises, valores 0-5)
    respetando el split oficial (fold1=train, fold2=val, fold3=test), a
    diferencia del split aleatorio 80/10/10 que usa 05 para DSB2018.

    Returns:
        Dict[str, Tuple[str, str]]: {"train": (images_dir, masks_dir), ...}
    """
    split_dirs = {}

    for fold_name, split_name in FOLD_TO_SPLIT.items():
        fold_dir = fold_paths[fold_name]
        images_npy = np.load(os.path.join(fold_dir, "images.npy"))
        masks_npy = np.load(os.path.join(fold_dir, "masks.npy"))

        images_out = os.path.join(output_dir, split_name, "images")
        masks_out = os.path.join(output_dir, split_name, "masks")
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(masks_out, exist_ok=True)

        print(f"[INFO] Estructurando {fold_name} -> '{split_name}' ({len(images_npy)} parches)...")
        for i in range(len(images_npy)):
            img = images_npy[i].astype(np.uint8)
            indexed_mask = convert_pannuke_mask_to_indexed(masks_npy[i])

            sample_name = f"{fold_name}_{i:05d}.png"
            Image.fromarray(img).save(os.path.join(images_out, sample_name))
            Image.fromarray(indexed_mask, mode="L").save(os.path.join(masks_out, sample_name))

        split_dirs[split_name] = (images_out, masks_out)
        print(f"  -> '{split_name}' listo en '{images_out}' / '{masks_out}'.")

    return split_dirs


# =============================================================================
# SECCION 3: VALIDACION + REPORTE DE DISTRIBUCION DE CLASES (DESBALANCE)
# =============================================================================

def validate_and_report_classes(split_dirs: Dict[str, Tuple[str, str]]) -> Dict[str, Any]:
    """
    Verifica emparejamiento imagen-mascara y reporta la distribucion de
    pixeles por clase en el split de entrenamiento, para detectar el
    desbalance entre clases (p. ej. 'Muerto' suele ser mucho mas raro que
    'Neoplasico').
    """
    print("\n" + "=" * 60)
    print(" REPORTE DE VALIDACION Y DISTRIBUCION DE CLASES (PanNuke)")
    print("=" * 60)

    class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    report: Dict[str, Any] = {}

    for split_name, (images_dir, masks_dir) in split_dirs.items():
        img_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        matched = len(img_paths) == len(mask_paths) and len(img_paths) > 0
        report[split_name] = {"num_samples": len(img_paths), "matched": matched}
        print(f"  - {split_name:5s}: {len(img_paths)} muestras | pares OK: {matched}")

        if split_name == "train":
            for mp in mask_paths:
                mask_arr = np.array(Image.open(mp))
                counts = np.bincount(mask_arr.ravel(), minlength=NUM_CLASSES)
                class_pixel_counts += counts[:NUM_CLASSES]

    total_px = class_pixel_counts.sum()
    print("\n  Distribucion de pixeles por clase (split train):")
    for idx, name in enumerate(CLASS_NAMES):
        pct = 100.0 * class_pixel_counts[idx] / total_px if total_px > 0 else 0.0
        print(f"    [{idx}] {name:12s}: {pct:6.2f}%")
    print("=" * 60 + "\n")

    report["class_pixel_counts"] = class_pixel_counts.tolist()
    return report


# =============================================================================
# SECCION 4: DATASET Y DATALOADERS MULTICLASE
# =============================================================================

class PanNukeSegmentationDataset(Dataset):
    """
    Analogo a `BioImageSegmentationDataset` (script 05), pero preserva las
    6 clases en vez de binarizar. La mascara se devuelve como LongTensor
    de indices de clase (H, W): el formato que espera `nn.CrossEntropyLoss`,
    no un FloatTensor (1, H, W) tipo `BCEWithLogitsLoss`.
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
        """Mismas transformaciones que 05: flips + rotaciones de 90 grados, sincronizadas."""
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
        mask_pil = Image.open(self.mask_paths[idx]).convert("L")  # 1 canal, valores 0-5

        img_pil = img_pil.resize(self.target_size, Image.BILINEAR)
        mask_pil = mask_pil.resize(self.target_size, Image.NEAREST)  # NEAREST: nunca inventar clases nuevas

        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        mask_np = np.array(mask_pil, dtype=np.int64)  # indices de clase, NO binarizar

        if self.transform:
            img_np, mask_np = self._apply_synchronous_transforms(img_np, mask_np)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W) float
        mask_tensor = torch.from_numpy(mask_np).long()          # (H, W) long -> listo para CrossEntropyLoss

        return img_tensor, mask_tensor


def create_pannuke_dataloaders(
    split_dirs: Dict[str, Tuple[str, str]],
    batch_size: int = 8,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    A diferencia de `create_dataloaders()` de 05 (split aleatorio 80/10/10),
    aqui el split ya viene fijado por los folds oficiales de PanNuke.
    """
    datasets = {}
    for split_name, (images_dir, masks_dir) in split_dirs.items():
        img_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        datasets[split_name] = PanNukeSegmentationDataset(
            img_paths, mask_paths, transform=(split_name == "train")
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
    print(" DLBA - SCRIPT 07: PREPROCESAMIENTO PANNUKE (MULTICLASE) ")
    print("=================================================================")

    fold_paths = locate_or_generate_raw_folds()
    split_dirs = structure_pannuke_folds(fold_paths)
    validate_and_report_classes(split_dirs)

    print("[INFO] Construyendo DataLoaders (split oficial fold1/fold2/fold3)...")
    train_loader, val_loader, test_loader = create_pannuke_dataloaders(split_dirs, batch_size=8)

    images, masks = next(iter(train_loader))
    print(f"\n[OK] Batch de entrenamiento cargado:")
    print(f"     Imagen : {tuple(images.shape)}, dtype={images.dtype}")
    print(f"     Mascara: {tuple(masks.shape)}, dtype={masks.dtype}, clases presentes: {torch.unique(masks).tolist()}")

    print("\n[OK] Script 07 completado.")
    print("     Para entrenar: UNet(in_channels=3, out_channels=6) + nn.CrossEntropyLoss()")
    print("     en vez de BCEWithLogitsLoss + DiceLoss binaria (ver 04_unet_segmentation.py).")