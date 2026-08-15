r"""
===============================================================================
MODULO 2 -- SEGMENTACION SEMANTICA BINARIA DE NUCLEOS SOBRE PanNuke (dataset/)
===============================================================================

CONTEXTO Y DECISIONES (ver bitacora_modulo2.txt para el detalle completo)
-------------------------------------------------------------------------------
El dataset elegido para el Seminario es PanNuke (linea "Datasets para
Segmentacion Semantica" de DATASETS_OVERVIEW.md), especificamente el paquete
de Kaggle "PanNuke dataset - experimental data"
(https://www.kaggle.com/datasets/theredlad/pannuke-dataset-experimental-data,
el mismo enlazado en DATASETS_OVERVIEW.md).

DESCARGA AUTOMATICA (sin dependencia de copia manual)
-------------------------------------------------------------------------------
Este script YA NO depende de que el dataset este copiado a mano en `dataset/`:
  1. Si `dataset/train/images` y `dataset/validate/images` ya existen y tienen
     archivos (por ejemplo, porque ya lo descargaste antes), se usan tal cual
     -- no se vuelve a descargar nada.
  2. Si no existen, el script descarga el dataset automaticamente desde
     Kaggle usando el CLI oficial `kaggle` (paquete 'kaggle', v2.x):
     `kaggle datasets download llwlabs/pannuke -p dataset --unzip`.

  Se uso el handle `llwlabs/pannuke` (en vez de `theredlad/pannuke-dataset-
  experimental-data`, usado en una version anterior de este script) tras
  verificar mediante `kaggle datasets files llwlabs/pannuke` que es el mismo
  dataset: mismo nombre y tamano exacto de `model.h5` (276,914,520 bytes,
  identico byte a byte al que ya esta en `dataset/`) y misma convencion de
  nombres de archivo (`train/images/{fold}-{indice}.png`).

La API de Kaggle SIEMPRE requiere una cuenta (no hay forma de saltarse esto,
ni con este script ni con ningun otro codigo que use la API de Kaggle), pero
es un paso de UNA SOLA VEZ, no una descarga manual del dataset en si:

  Opcion recomendada -- login por navegador (sin manejar archivos a mano):
      kaggle auth login
    Abre el navegador, inicias sesion con tu cuenta de Kaggle, y las
    credenciales quedan cacheadas localmente.

  Alternativa "legacy" con token manual:
    1. Crea una cuenta gratuita en https://www.kaggle.com
    2. Ve a https://www.kaggle.com/settings/api -> "Create New Token"
    3. Sigue las instrucciones que imprime `kaggle --help` para guardar el
       token (variable de entorno KAGGLE_API_TOKEN o archivo
       ~/.kaggle/access_token; versiones previas del CLI usaban
       ~/.kaggle/kaggle.json con usuario+key).

Instalar el CLI: pip install kaggle

DISPOSITIVO: PRIORIDAD GPU, FALLBACK A CPU
-------------------------------------------------------------------------------
El script detecta automaticamente si hay una GPU compatible con CUDA
disponible (`torch.cuda.is_available()`) y la usa por defecto; si no
encuentra ninguna, cae automaticamente a CPU sin intervencion manual. Se
puede forzar un dispositivo especifico con `--device cuda` o `--device cpu`.

Se verifico (muestreo aleatorio de ~600 mascaras, todas con valores unicos
{0, 255}) que esta copia YA VIENE COLAPSADA a segmentacion BINARIA
(nucleo=255 vs. fondo=0): la informacion de las 5 categorias de nucleo del
PanNuke original (Neoplasico, Inflamatorio, Conectivo, Muerto, Epitelial) no
esta presente en estos PNG. El `model.h5` incluido confirma lo mismo: su
ultima capa es Conv2D(filters=1, activation='sigmoid'), es decir, un U-Net
binario. Por eso este entregable implementa Segmentacion Binaria (Opcion A),
no multiclase.

ESTRUCTURA REAL DEL DATASET (verificada, no asumida)
-------------------------------------------------------------------------------
dataset/
|-- model.h5                  <- baseline Keras (U-Net binario 256x256x3->256x256x1)
|-- train/{images,masks}/     <- 6716 pares PNG (256x256 RGB / L binaria)
|-- validate/{images,masks}/  <- 1185 pares PNG (256x256 RGB / L binaria)

Los nombres de archivo siguen el patron "{fold}-{indice}.png" (fold en
{1,2,3}), correspondiente a los 3 folds originales de PanNuke ya mezclados
por el autor del paquete de Kaggle (6716 + 1185 = 7901 = total oficial de
PanNuke).

PIPELINE DE ESTE SCRIPT
-------------------------------------------------------------------------------
1. Carga `train/` y `validate/` reutilizando `BioImageSegmentationDataset`
   (script 05, importada dinamicamente) sin modificarla -- ya asume mascaras
   binarias con el mismo formato que aqui se confirmo.
2. Reconstruye la arquitectura U-Net y la DiceLoss de `04_unet_segmentation.py`
   de forma autocontenida (evita forzar la importacion de `napari`, que 04
   requiere a nivel de modulo aunque no se use la visualizacion interactiva).
3. Entrena con BCEWithLogitsLoss + DiceLoss (Adam), valida cada epoca.
4. Evalua en el set de prueba (`validate/`) con Dice e IoU.
5. (Opcional) Compara contra el baseline `dataset/model.h5` si TensorFlow
   esta disponible en el entorno.
6. Guarda curvas de entrenamiento, ejemplos cualitativos y un resumen de
   metricas (JSON) en `outputs_modulo2_pannuke/`.

USO
-------------------------------------------------------------------------------
    python modulo2_pannuke_binary_segmentation.py --epochs 15 --batch-size 8

    # corrida rapida de verificacion (smoke test):
    python modulo2_pannuke_binary_segmentation.py --max-train-samples 32 \
        --max-test-samples 16 --epochs 1
===============================================================================
"""

import os
import sys
import glob
import json
import time
import random
import shutil
import argparse
import importlib
import subprocess
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # backend sin GUI: seguro para correr en background/servidor/Colab
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "dataset")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_modulo2_pannuke")
KAGGLE_DATASET_HANDLE = "llwlabs/pannuke"


# =============================================================================
# PARTE 0: DESCARGA AUTOMATICA DEL DATASET (Kaggle) Y SELECCION DE DISPOSITIVO
# =============================================================================

def _has_expected_structure(path: str) -> bool:
    """Verifica que `path` tenga train/images y validate/images con archivos."""
    train_images = os.path.join(path, "train", "images")
    val_images = os.path.join(path, "validate", "images")
    if not (os.path.isdir(train_images) and os.path.isdir(val_images)):
        return False
    return len(os.listdir(train_images)) > 0 and len(os.listdir(val_images)) > 0


def resolve_dataset_dir(data_dir: str) -> str:
    """Devuelve `data_dir` con la estructura esperada
    (train/{images,masks}, validate/{images,masks}[, model.h5]), descargando
    el dataset desde Kaggle automaticamente (CLI oficial `kaggle`) si no se
    encuentra ya ahi.

    No depende de una copia manual: si `data_dir` ya tiene los datos, se
    reutilizan tal cual (sin descargar de nuevo); si no, se descargan
    directamente dentro de `data_dir` con:
        kaggle datasets download <KAGGLE_DATASET_HANDLE> -p <data_dir> --unzip
    """
    if _has_expected_structure(data_dir):
        print(f"[INFO] Dataset ya presente localmente en '{data_dir}' -- se omite la descarga.")
        return data_dir

    print(
        f"[INFO] No se encontro el dataset en '{data_dir}'. "
        f"Descargando automaticamente desde Kaggle ('{KAGGLE_DATASET_HANDLE}') via el CLI oficial..."
    )

    kaggle_exe = shutil.which("kaggle")
    if kaggle_exe is None:
        raise RuntimeError(
            "No se encontro el ejecutable 'kaggle' en el PATH.\n"
            "Instala el CLI oficial con:\n    pip install kaggle\n"
            "y asegurate de correr este script en el mismo entorno/terminal donde lo instalaste."
        )

    os.makedirs(data_dir, exist_ok=True)

    # En Windows, subprocess.run(["kaggle", ...]) sin resolver la ruta completa
    # primero puede fallar con FileNotFoundError aunque 'kaggle' funcione bien
    # escrito directo en la terminal -- por eso se usa kaggle_exe (ruta completa
    # resuelta por shutil.which) en vez del string "kaggle" suelto.
    resultado = subprocess.run(
        [kaggle_exe, "datasets", "download", KAGGLE_DATASET_HANDLE, "-p", data_dir, "--unzip"],
        capture_output=True, text=True,
    )
    salida = (resultado.stdout + resultado.stderr).strip()

    if resultado.returncode != 0:
        auth_hint = ""
        if "403" in salida or "forbidden" in salida.lower() or "auth" in salida.lower():
            auth_hint = (
                "\n\nEsto parece un problema de autenticacion (paso de UNA SOLA VEZ, no "
                "requiere descargar el dataset a mano). Corre esto en la terminal y vuelve "
                "a ejecutar este script:\n"
                "    kaggle auth login\n"
                "(alternativa sin navegador: token clasico -- ver docstring de este script "
                "o `kaggle --help`)."
            )
        raise RuntimeError(f"La descarga de Kaggle fallo:\n{salida}{auth_hint}")

    if not _has_expected_structure(data_dir):
        raise RuntimeError(
            f"La descarga se completo en '{data_dir}' pero no tiene la estructura esperada "
            "(train/images, validate/images). Revisa el contenido actual del dataset en "
            f"https://www.kaggle.com/datasets/{KAGGLE_DATASET_HANDLE}"
        )

    print(f"[OK] Dataset descargado y disponible en: {data_dir}")
    return data_dir


def select_device(preferred: str = "auto") -> torch.device:
    """Prioriza GPU (CUDA); si no hay ninguna disponible, usa CPU automaticamente."""
    if preferred == "cpu":
        print("[INFO] Dispositivo forzado por --device: CPU.")
        return torch.device("cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda fue solicitado pero no hay GPU CUDA disponible.")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INFO] GPU detectada ({gpu_name}) -- se usara CUDA.")
        return torch.device("cuda")

    print("[INFO] No se detecto GPU compatible con CUDA -- se usara CPU.")
    return torch.device("cpu")


# =============================================================================
# PARTE 1: ARQUITECTURA U-NET Y DICE LOSS
#   Replica fiel de las clases definidas en 04_unet_segmentation.py.
# =============================================================================

class DoubleConv(nn.Module):
    """[Conv3x3 -> BatchNorm -> ReLU] x2, con padding=1 (mantiene H, W)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """MaxPool(2x2) -> DoubleConv. Encoder: reduce H,W a la mitad, duplica canales."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """ConvTranspose(2x2) -> concat skip connection -> DoubleConv. Decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net binaria: Encoder (4 niveles) + Bottleneck + Decoder con skip connections."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class DiceLoss(nn.Module):
    """Dice Loss binaria: 1 - (2*interseccion)/(union), sobre probabilidades sigmoid."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice_coeff


# =============================================================================
# PARTE 2: DATASET -- reutiliza BioImageSegmentationDataset de 05 sin tocarla
# =============================================================================

def _load_bioimage_dataset_class():
    """Importa dinamicamente 05_dataset_exploration_and_dataloaders.py (nombre
    de modulo con digito inicial: no es importable con `import`, solo via
    importlib -- mismo patron ya usado en 06_dataset_download_and_structuring.py."""
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    module = importlib.import_module("05_dataset_exploration_and_dataloaders")
    return module.BioImageSegmentationDataset


def build_datasets(
    data_dir: str,
    seed: int = 42,
    val_fraction: float = 0.1,
    max_train_samples: int = None,
    max_test_samples: int = None,
):
    """Construye train/val/test respetando la particion real del dataset:
    - train/  -> se reparte internamente en train (90%) / val (10%) para
                 monitorear el entrenamiento.
    - validate/ -> se reserva integro como test final (nunca visto en training).
    """
    BioImageSegmentationDataset = _load_bioimage_dataset_class()

    train_imgs = sorted(glob.glob(os.path.join(data_dir, "train", "images", "*.png")))
    train_masks = sorted(glob.glob(os.path.join(data_dir, "train", "masks", "*.png")))
    test_imgs = sorted(glob.glob(os.path.join(data_dir, "validate", "images", "*.png")))
    test_masks = sorted(glob.glob(os.path.join(data_dir, "validate", "masks", "*.png")))

    assert len(train_imgs) == len(train_masks) and len(train_imgs) > 0, (
        f"No se encontraron pares imagen/mascara en {data_dir}/train"
    )
    assert len(test_imgs) == len(test_masks) and len(test_imgs) > 0, (
        f"No se encontraron pares imagen/mascara en {data_dir}/validate"
    )

    n_train_pool_total = len(train_imgs)
    n_test_pool_total = len(test_imgs)

    rng = random.Random(seed)

    train_pairs = list(zip(train_imgs, train_masks))
    rng.shuffle(train_pairs)
    if max_train_samples is not None:
        train_pairs = train_pairs[:max_train_samples]

    n_val = max(1, int(round(val_fraction * len(train_pairs))))
    val_pairs = train_pairs[:n_val]
    tr_pairs = train_pairs[n_val:]

    test_pairs = list(zip(test_imgs, test_masks))
    rng.shuffle(test_pairs)
    if max_test_samples is not None:
        test_pairs = test_pairs[:max_test_samples]

    def unzip(pairs):
        if not pairs:
            return [], []
        imgs, masks = zip(*pairs)
        return list(imgs), list(masks)

    tr_imgs, tr_masks = unzip(tr_pairs)
    va_imgs, va_masks = unzip(val_pairs)
    te_imgs, te_masks = unzip(test_pairs)

    train_ds = BioImageSegmentationDataset(tr_imgs, tr_masks, transform=True)
    val_ds = BioImageSegmentationDataset(va_imgs, va_masks, transform=False)
    test_ds = BioImageSegmentationDataset(te_imgs, te_masks, transform=False)

    stats = {
        "n_train_pool_total": n_train_pool_total,
        "n_test_pool_total": n_test_pool_total,
        "n_train_used": len(tr_imgs),
        "n_val_used": len(va_imgs),
        "n_test_used": len(te_imgs),
    }
    return train_ds, val_ds, test_ds, stats, (te_imgs, te_masks)


# =============================================================================
# PARTE 3: METRICAS (Dice coefficient / IoU-Jaccard)
# =============================================================================

def dice_coefficient(pred_bin: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    dims = tuple(range(1, pred_bin.ndim))
    intersection = (pred_bin * target).sum(dim=dims)
    union = pred_bin.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * intersection + eps) / (union + eps)


def iou_score(pred_bin: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    dims = tuple(range(1, pred_bin.ndim))
    intersection = (pred_bin * target).sum(dim=dims)
    union = ((pred_bin + target) > 0).float().sum(dim=dims)
    return (intersection + eps) / (union + eps)


# =============================================================================
# PARTE 4: ENTRENAMIENTO Y EVALUACION
# =============================================================================

def run_epoch(model, loader, device, optimizer, criterion_bce, criterion_dice, train_mode: bool):
    model.train() if train_mode else model.eval()
    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            if train_mode:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion_bce(logits, masks) + criterion_dice(logits, masks)

            if train_mode:
                loss.backward()
                optimizer.step()

            batch_n = images.size(0)
            total_loss += loss.item() * batch_n
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                total_dice += dice_coefficient(preds, masks).sum().item()
                total_iou += iou_score(preds, masks).sum().item()
            n += batch_n

    return {"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n}


def train_model(model, train_loader, val_loader, device, epochs, lr, output_dir):
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_val_dice = -1.0
    best_ckpt_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, optimizer, criterion_bce, criterion_dice, True)
        val_metrics = run_epoch(model, val_loader, device, optimizer, criterion_bce, criterion_dice, False)
        dt = time.time() - t0

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} ({dt:.1f}s)"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), best_ckpt_path)

    return history, best_ckpt_path


def evaluate_model(model, test_loader, device):
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    return run_epoch(model, test_loader, device, None, criterion_bce, criterion_dice, False)


# =============================================================================
# PARTE 5: COMPARACION OPCIONAL CONTRA EL BASELINE dataset/model.h5 (Keras)
# =============================================================================

def evaluate_keras_baseline(model_h5_path: str, test_img_paths, test_mask_paths, image_size: int = 256):
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        print(
            "[INFO] TensorFlow no esta instalado -- se omite la comparacion con el baseline "
            f"'{model_h5_path}'. Instala con `pip install tensorflow` para habilitarla."
        )
        return None

    if not os.path.exists(model_h5_path):
        print(f"[WARN] No se encontro el baseline en '{model_h5_path}'. Se omite la comparacion.")
        return None

    from PIL import Image

    keras_model = load_model(model_h5_path)
    dice_total, iou_total, n = 0.0, 0.0, 0
    eps = 1e-7

    for img_path, mask_path in zip(test_img_paths, test_mask_paths):
        img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
        mask = Image.open(mask_path).convert("L").resize((image_size, image_size))
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)

        pred = keras_model.predict(img_arr[None, ...], verbose=0)[0, ..., 0]
        pred_bin = (pred > 0.5).astype(np.float32)

        intersection = float((pred_bin * mask_arr).sum())
        dice = (2.0 * intersection + eps) / (pred_bin.sum() + mask_arr.sum() + eps)
        union = float(((pred_bin + mask_arr) > 0).sum())
        iou = (intersection + eps) / (union + eps)

        dice_total += dice
        iou_total += iou
        n += 1

    return {"dice": dice_total / n, "iou": iou_total / n, "n_samples": n}


# =============================================================================
# PARTE 6: VISUALIZACION (figuras estaticas -- seguras para correr headless)
# =============================================================================

def save_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss (BCE + Dice)")
    axes[0].set_title("Curva de perdida")
    axes[0].legend()

    axes[1].plot(epochs, history["val_dice"], label="Val Dice")
    axes[1].plot(epochs, history["val_iou"], label="Val IoU")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Metricas de validacion")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_qualitative_predictions(model, test_dataset, device, output_dir, n_samples: int = 4):
    n_samples = min(n_samples, len(test_dataset))
    if n_samples == 0:
        return None

    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3.2 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            img_t, mask_t = test_dataset[i]
            logits = model(img_t.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_bin = (prob > 0.5).astype(np.float32)

            img_np = img_t.permute(1, 2, 0).numpy()
            mask_np = mask_t.squeeze().numpy()

            axes[i, 0].imshow(img_np); axes[i, 0].set_title("Imagen H&E"); axes[i, 0].axis("off")
            axes[i, 1].imshow(mask_np, cmap="gray"); axes[i, 1].set_title("Mascara real"); axes[i, 1].axis("off")
            axes[i, 2].imshow(prob, cmap="magma"); axes[i, 2].set_title("Probabilidad U-Net"); axes[i, 2].axis("off")
            axes[i, 3].imshow(pred_bin, cmap="gray"); axes[i, 3].set_title("Prediccion binaria"); axes[i, 3].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "qualitative_predictions.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Modulo 2 - Segmentacion binaria de nucleos (PanNuke)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help="'auto' (por defecto) prioriza GPU y usa CPU si no hay ninguna disponible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    print(f"[INFO] Dispositivo seleccionado: {device}")

    args.data_dir = resolve_dataset_dir(args.data_dir)
    print(f"[INFO] Dataset: {args.data_dir}")

    train_ds, val_ds, test_ds, stats, (test_img_paths, test_mask_paths) = build_datasets(
        args.data_dir,
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    print(
        f"[INFO] Muestras -> train={stats['n_train_used']} val={stats['n_val_used']} "
        f"test={stats['n_test_used']} "
        f"(pool total en disco: train/={stats['n_train_pool_total']} validate/={stats['n_test_pool_total']})"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] U-Net instanciada -- {n_params:,} parametros entrenables")

    t_start = time.time()
    history, best_ckpt_path = train_model(
        model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, output_dir=args.output_dir
    )
    train_time = time.time() - t_start
    print(f"[INFO] Entrenamiento completo en {train_time / 60:.1f} min. Mejor checkpoint: {best_ckpt_path}")

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_metrics = evaluate_model(model, test_loader, device)
    print(
        f"[RESULT] Test -- Dice={test_metrics['dice']:.4f} IoU={test_metrics['iou']:.4f} "
        f"(n={stats['n_test_used']})"
    )

    curves_path = save_training_curves(history, args.output_dir)
    quali_path = save_qualitative_predictions(model, test_ds, device, args.output_dir)
    print(f"[INFO] Figuras guardadas: {curves_path}, {quali_path}")

    baseline_metrics = None
    if not args.skip_baseline:
        baseline_metrics = evaluate_keras_baseline(
            os.path.join(args.data_dir, "model.h5"), test_img_paths, test_mask_paths
        )
        if baseline_metrics:
            print(
                f"[RESULT] Baseline model.h5 -- Dice={baseline_metrics['dice']:.4f} "
                f"IoU={baseline_metrics['iou']:.4f} (n={baseline_metrics['n_samples']})"
            )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "hyperparameters": vars(args),
        "dataset_stats": stats,
        "n_trainable_params": n_params,
        "train_time_minutes": round(train_time / 60, 2),
        "history": history,
        "test_metrics": test_metrics,
        "baseline_model_h5_metrics": baseline_metrics,
    }
    summary_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Resumen de metricas guardado en: {summary_path}")
    print("\n[OK] Modulo 2 -- Segmentacion binaria PanNuke: pipeline completo ejecutado.")


if __name__ == "__main__":
    main()
