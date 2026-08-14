"""
===============================================================================
SCRIPT: DESCARGA AUTOMATICA DEL DATASET PANNUKE DESDE KAGGLE
===============================================================================
Descarga el dataset "PanNuke Dataset (Experimental Data)" de Kaggle
(https://www.kaggle.com/datasets/theredlad/pannuke-dataset-experimental-data),
lo descomprime y reorganiza los archivos en la estructura que espera
`preprosesamiento.py`:

    data/raw/pannuke_source/fold1/{images.npy, masks.npy, types.npy}
    data/raw/pannuke_source/fold2/{images.npy, masks.npy, types.npy}
    data/raw/pannuke_source/fold3/{images.npy, masks.npy, types.npy}

REQUISITOS:
  1. pip install kaggle
  2. Completa tus credenciales de Kaggle en KAGGLE_USERNAME y KAGGLE_KEY
     mas abajo (Settings -> API -> "Create New Token" en kaggle.com genera
     un kaggle.json con ambos valores).

USO:
  python descargar_pannuke.py
===============================================================================
"""

import os
import re
import shutil
import zipfile

# =============================================================================
# CREDENCIALES DE KAGGLE (COMPLETAR ANTES DE EJECUTAR)
# =============================================================================
KAGGLE_USERNAME = ""
KAGGLE_KEY = ""

DATASET_SLUG = "theredlad/pannuke-dataset-experimental-data"
DOWNLOAD_DIR = "data/raw/pannuke_download"
EXTRACT_DIR = "data/raw/pannuke_extracted"
RAW_DATA_DIR = "data/raw/pannuke_source"


def _configurar_credenciales() -> None:
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        raise RuntimeError(
            "Completa KAGGLE_USERNAME y KAGGLE_KEY al inicio de este script "
            "con los datos de tu token de Kaggle (Account -> Create New Token)."
        )
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY


def descargar_dataset() -> str:
    """Descarga el .zip del dataset usando la API de Kaggle. Devuelve la ruta del .zip."""
    _configurar_credenciales()
    from kaggle.api.kaggle_api_extended import KaggleApi  # import tardio: necesita las env vars ya seteadas

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"[INFO] Descargando '{DATASET_SLUG}' desde Kaggle...")
    api.dataset_download_files(DATASET_SLUG, path=DOWNLOAD_DIR, unzip=False, quiet=False)

    zip_candidates = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".zip")]
    if not zip_candidates:
        raise RuntimeError(f"No se encontro ningun .zip descargado en '{DOWNLOAD_DIR}'.")

    zip_path = os.path.join(DOWNLOAD_DIR, zip_candidates[0])
    print(f"[OK] Descargado: {zip_path}")
    return zip_path


def descomprimir(zip_path: str) -> None:
    print(f"[INFO] Descomprimiendo en '{EXTRACT_DIR}'...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print("[OK] Descompresion completa.")


def reorganizar_folds() -> None:
    """
    El .zip oficial de PanNuke trae cada fold en subcarpetas anidadas del
    estilo 'Fold 1/images/fold1/images.npy', 'Fold 1/masks/fold1/masks.npy',
    etc. Aqui se localizan los 3 archivos .npy de cada fold (sin importar
    como esten anidados) y se copian a la estructura plana
    data/raw/pannuke_source/foldN/ que espera preprosesamiento.py.
    """
    print(f"[INFO] Reorganizando folds en '{RAW_DATA_DIR}'...")
    fold_pattern = re.compile(r"fold[\s_-]?([123])", re.IGNORECASE)

    encontrados = {"1": {}, "2": {}, "3": {}}
    for root, _dirs, files in os.walk(EXTRACT_DIR):
        for filename in files:
            if filename not in ("images.npy", "masks.npy", "types.npy"):
                continue
            full_path = os.path.join(root, filename)
            match = fold_pattern.search(full_path)
            if not match:
                continue
            fold_num = match.group(1)
            clave = filename.replace(".npy", "")
            encontrados[fold_num][clave] = full_path

    for fold_num, archivos in encontrados.items():
        faltantes = {"images", "masks", "types"} - archivos.keys()
        if faltantes:
            print(f"[!] fold{fold_num}: faltan archivos {faltantes}, se omite.")
            continue

        fold_dir = os.path.join(RAW_DATA_DIR, f"fold{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        for clave, origen in archivos.items():
            destino = os.path.join(fold_dir, f"{clave}.npy")
            shutil.copyfile(origen, destino)
        print(f"  -> fold{fold_num} listo en '{fold_dir}'.")


if __name__ == "__main__":
    zip_path = descargar_dataset()
    descomprimir(zip_path)
    reorganizar_folds()
    print("\n[OK] Dataset PanNuke listo. Ahora puedes correr:")
    print("     python preprosesamiento.py")
