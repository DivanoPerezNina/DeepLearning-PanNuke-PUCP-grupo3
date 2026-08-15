"""
===============================================================================
DESCARGA DEL DATASET PANNUKE DESDE KAGGLE (llwlabs/pannuke)
===============================================================================
Descarga el mirror de PanNuke usando el CLI oficial de Kaggle (paquete
'kaggle', v2.x). A diferencia de 06_dataset_download_and_structuring.py
(que usa pooch para descargas publicas sin autenticacion), Kaggle requiere
una cuenta.

REQUISITO PREVIO -- autenticarte con Kaggle (una sola vez, desde la terminal):

    kaggle auth login

Esto abre el navegador para iniciar sesion con tu cuenta de Kaggle y guarda
las credenciales localmente; no hace falta manejar ningun archivo a mano.
(El metodo clasico con token -- Kaggle -> Settings -> API -> "Create New
Token", guardado como kaggle.json -- sigue funcionando como alternativa
"Legacy" si el login por navegador no te sirve.)

Instalar el paquete: pip install kaggle

NOTA sobre el notebook de Kaggle que mencionaste: si ese notebook corre
DENTRO de Kaggle (un "Kaggle Notebook"/kernel), ahi los datos ya vienen
montados en /kaggle/input/ sin necesidad de descargar nada -- ese patron
no aplica en tu maquina local. Este script es el equivalente para correr
fuera de Kaggle, en tu propio entorno.
===============================================================================
"""

import os
import shutil
import subprocess
import sys

DATASET_SLUG = "llwlabs/pannuke"
DOWNLOAD_DIR = "data/raw_download"


def descargar_dataset(dataset_slug: str = DATASET_SLUG, destino: str = DOWNLOAD_DIR) -> bool:
    """Descarga y descomprime el dataset de Kaggle via el CLI. Devuelve
    True si salio bien, False si fallo (verificado: Kaggle responde con
    '403 Forbidden' cuando falta autenticacion, no un error de red)."""
    kaggle_exe = shutil.which("kaggle")
    if kaggle_exe is None:
        print("[!] No se encontro el ejecutable 'kaggle' en el PATH.")
        print("[INFO] Verifica que 'pip install kaggle' haya terminado sin errores,")
        print("[INFO] y que estes en el mismo entorno/terminal donde lo instalaste.")
        return False

    os.makedirs(destino, exist_ok=True)
    print(f"[INFO] Descargando '{dataset_slug}' hacia '{destino}'...")

    # En Windows, subprocess.run(["kaggle", ...]) sin resolver la ruta completa
    # primero puede fallar con FileNotFoundError aunque 'kaggle' funcione bien
    # escrito directo en la terminal -- por eso se usa kaggle_exe (ruta completa
    # resuelta por shutil.which) en vez del string "kaggle" suelto.
    resultado = subprocess.run(
        [kaggle_exe, "datasets", "download", dataset_slug, "-p", destino, "--unzip"],
        capture_output=True, text=True,
    )
    salida = (resultado.stdout + resultado.stderr).strip()

    if resultado.returncode != 0:
        print("[!] La descarga fallo.")
        print(" ", salida)
        if "403" in salida or "forbidden" in salida.lower() or "auth" in salida.lower():
            print()
            print("[INFO] Es un problema de autenticacion. Corre esto una sola")
            print("[INFO] vez en la terminal y vuelve a correr este script:")
            print("[INFO]     kaggle auth login")
        return False

    print(f"[OK] Dataset descargado y descomprimido en '{destino}'.")
    return True


def listar_contenido(destino: str = DOWNLOAD_DIR, max_items: int = 20) -> None:
    """Muestra el contenido de primer nivel del resultado, para confirmar
    a simple vista si la estructura es train/validate con images/masks."""
    if not os.path.isdir(destino):
        return
    print(f"\n[INFO] Contenido de '{destino}' (primeros {max_items}):")
    for i, nombre in enumerate(sorted(os.listdir(destino))):
        if i >= max_items:
            print("  ...")
            break
        ruta = os.path.join(destino, nombre)
        tipo = "carpeta" if os.path.isdir(ruta) else "archivo"
        print(f"  [{tipo}] {nombre}")


if __name__ == "__main__":
    print("=================================================================")
    print(" DESCARGA DE PANNUKE (llwlabs/pannuke) DESDE KAGGLE ")
    print("=================================================================")

    ok = descargar_dataset()
    if not ok:
        sys.exit(1)

    listar_contenido()

    print("\n[OK] Listo. Compara esta estructura contra lo que espera")
    print("     preprocesamiento_binario.py (train/images, train/masks,")
    print("     validate/images, validate/masks) y ajusta rutas si no calzan.")