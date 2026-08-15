"""
===============================================================================
ENTRENAMIENTO: U-NET PARA SEGMENTACION BINARIA DE NUCLEOS (PANNUKE)
===============================================================================
Entrena la UNet(in_channels=3, out_channels=1) de 04_unet_segmentation.py
sobre los DataLoaders binarios que arma preprocesamiento_binario.py.

Perdida: BCEWithLogitsLoss -- coincide con out_channels=1: la UNet devuelve
logits crudos (sin sigmoid propio), y BCEWithLogitsLoss ya aplica el sigmoid
internamente antes de calcular la perdida (mas estable numericamente que
aplicar sigmoid aparte y despues BCE normal).

Metricas: Dice y IoU, tal como pide el Modulo 2 del curso (DATASETS_OVERVIEW.md).

Requiere que 04_unet_segmentation.py y preprocesamiento_binario.py esten en
la misma carpeta que este archivo.
===============================================================================
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# 04_unet_segmentation.py empieza con "04_", un nombre invalido para un
# import normal de Python -- ademas ese archivo importa napari a nivel de
# modulo, que puede no estar instalado o no tener GUI disponible. Por eso
# se extrae solo el codigo de las clases del modelo (UNet y sus bloques),
# sin ejecutar el resto del archivo.
_codigo_04 = (Path(__file__).parent / "04_unet_segmentation.py").read_text(encoding="utf-8")
_inicio = _codigo_04.index("class DoubleConv")
_fin = _codigo_04.index("class DiceLoss")
_namespace = {"nn": nn, "torch": torch, "F": torch.nn.functional}
exec(_codigo_04[_inicio:_fin], _namespace)
UNet = _namespace["UNet"]

from preprocesamiento_binario import split_validate_into_val_test, create_pannuke_dataloaders

RUTA_MEJOR_MODELO = "mejor_modelo_pannuke.pth"


# =============================================================================
# METRICAS
# =============================================================================

def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Coeficiente de Dice sobre la prediccion binarizada en 0.5."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    interseccion = (preds * target).sum()
    union = preds.sum() + target.sum()
    return ((2 * interseccion + eps) / (union + eps)).item()


def iou_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Interseccion sobre Union (Jaccard) sobre la prediccion binarizada en 0.5."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    interseccion = (preds * target).sum()
    union = ((preds + target) > 0).float().sum()
    return ((interseccion + eps) / (union + eps)).item()


# =============================================================================
# UN EPOCH (ENTRENAMIENTO O VALIDACION/PRUEBA)
# =============================================================================

def correr_epoch(model, loader, criterion, optimizer, device, entrenando: bool):
    model.train() if entrenando else model.eval()
    perdida_total = dice_total = iou_total = 0.0
    n_batches = 0

    with torch.set_grad_enabled(entrenando):
        for imagenes, mascaras in loader:
            imagenes, mascaras = imagenes.to(device), mascaras.to(device)

            logits = model(imagenes)
            perdida = criterion(logits, mascaras)

            if entrenando:
                optimizer.zero_grad()
                perdida.backward()
                optimizer.step()

            perdida_total += perdida.item()
            dice_total += dice_score(logits, mascaras)
            iou_total += iou_score(logits, mascaras)
            n_batches += 1

    return perdida_total / n_batches, dice_total / n_batches, iou_total / n_batches


# =============================================================================
# SELECCION DE DISPOSITIVO (CON RESPALDO A CPU)
# =============================================================================

def obtener_device() -> torch.device:
    """No solo revisa torch.cuda.is_available() -- a veces eso da True pero
    la GPU falla al primer uso real (drivers desactualizados, o VRAM
    insuficiente incluso para una asignacion minima). Por eso se hace una
    prueba real y chica antes de comprometerse a usar CUDA."""
    if not torch.cuda.is_available():
        print("[INFO] No se detecto GPU con CUDA -- entrenando en CPU (va a ser mas lento).")
        print("[INFO] Si tienes GPU NVIDIA y esperabas que la usara, probablemente instalaste")
        print("[INFO] la version CPU-only de PyTorch por error -- revisa el comando exacto")
        print("[INFO] para tu sistema en https://pytorch.org/get-started/locally/")
        return torch.device("cpu")

    try:
        _prueba = torch.zeros(1).cuda()
        del _prueba
        torch.cuda.empty_cache()
    except RuntimeError as err:
        print(f"[!] Se detecto una GPU pero fallo al usarla ({err}).")
        print("[INFO] Cayendo a CPU en vez de interrumpir el script.")
        return torch.device("cpu")

    nombre_gpu = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[INFO] Usando GPU: {nombre_gpu} ({vram_gb:.1f} GB VRAM)")
    return torch.device("cuda")


# =============================================================================
# MAIN
# =============================================================================

def _entrenar(device: torch.device, epocas: int, batch_size: int, lr: float) -> None:
    """Cuerpo real del entrenamiento para un device dado. Separado de main()
    para poder reintentar entero en CPU si la GPU se queda sin memoria a
    mitad de camino (ver main())."""
    print("\n[INFO] === Paso 1/2: preprocesamiento (split de test + DataLoaders) ===")
    split_validate_into_val_test()
    train_loader, val_loader, test_loader = create_pannuke_dataloaders(batch_size=batch_size)
    print("[OK] Preprocesamiento listo -- DataLoaders de train/val/test armados.")

    print("\n[INFO] === Paso 2/2: entrenamiento ===")
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mejor_dice_val = 0.0
    print(f"\n[INFO] Entrenando {epocas} epocas (batch_size={batch_size}, lr={lr}, device={device})...\n")

    for epoca in range(1, epocas + 1):
        inicio = time.time()
        loss_tr, dice_tr, iou_tr = correr_epoch(model, train_loader, criterion, optimizer, device, entrenando=True)
        loss_val, dice_val, iou_val = correr_epoch(model, val_loader, criterion, optimizer, device, entrenando=False)
        duracion = time.time() - inicio

        print(f"[Epoca {epoca:2d}/{epocas}] "
              f"train: loss={loss_tr:.4f} dice={dice_tr:.4f} iou={iou_tr:.4f} | "
              f"val: loss={loss_val:.4f} dice={dice_val:.4f} iou={iou_val:.4f} "
              f"({duracion:.1f}s)")

        if dice_val > mejor_dice_val:
            mejor_dice_val = dice_val
            torch.save(model.state_dict(), RUTA_MEJOR_MODELO)
            print(f"       -> nuevo mejor Dice de validacion ({dice_val:.4f}), modelo guardado en '{RUTA_MEJOR_MODELO}'")

    print("\n[INFO] Evaluando en test con el mejor modelo guardado...")
    model.load_state_dict(torch.load(RUTA_MEJOR_MODELO, map_location=device))
    loss_test, dice_test, iou_test = correr_epoch(model, test_loader, criterion, optimizer, device, entrenando=False)
    print(f"[OK] Resultado final en test: loss={loss_test:.4f} dice={dice_test:.4f} iou={iou_test:.4f}")


def main(epocas: int = 15, batch_size: int = 8, lr: float = 1e-3):
    device = obtener_device()

    try:
        _entrenar(device, epocas, batch_size, lr)
    except RuntimeError as err:
        sin_memoria = "out of memory" in str(err).lower()
        if sin_memoria and device.type == "cuda":
            print(f"\n[!] La GPU se quedo sin memoria durante el entrenamiento: {err}")
            print("[INFO] Reintentando el entrenamiento completo en CPU (mas lento, pero termina).")
            print("[INFO] Si quieres seguir usando GPU, prueba de nuevo con un batch_size menor.")
            torch.cuda.empty_cache()
            _entrenar(torch.device("cpu"), epocas, batch_size, lr)
        else:
            raise


if __name__ == "__main__":
    main()
