# Resultados — Módulo 2 (segmentación binaria de núcleos, U-Net)

Esta carpeta contiene los **artefactos ya generados** de una corrida completa de
[`modulo2_pannuke_binary_segmentation.py`](../modulo2_pannuke_binary_segmentation.py)
(15 épocas, ~105 min en GPU). No hace falta volver a entrenar para revisarlos.

## Contenido

| Archivo | Qué es |
|---|---|
| `metrics_summary.json` | Dice/IoU finales sobre el set de test, tiempo de entrenamiento, device usado |
| `training_history.png` | Curva de pérdida/métrica por época |
| `qualitative_predictions.png` | Ejemplos: imagen H&E, máscara real, probabilidad predicha, predicción binaria |
| `notebook1445a8ba31.ipynb` | Notebook (pensado para Kaggle) que corrió el script y volcó estas salidas |

Resultado de esta corrida: **Dice 0.8283 / IoU 0.7228** (ver `metrics_summary.json`).
La comparación contra el baseline `model.h5` se omitió por incompatibilidad de
`Conv2DTranspose` entre el `.h5` original y Keras 3.

## Cómo ver los resultados (sin ejecutar nada)

Los PNG y el JSON son archivos estáticos, ábrelos directo:

```bash
start Resultados-Modulo2\training_history.png
start Resultados-Modulo2\qualitative_predictions.png
```

`metrics_summary.json` se abre con cualquier editor de texto.

## Sobre napari: no hace falta para esto

`environment.yml` instala `napari[all]`, pero **no es un requisito para ver ni para
generar estos resultados**. napari solo se usa dentro de
[`05_dataset_exploration_and_dataloaders.py`](../05_dataset_exploration_and_dataloaders.py)
(función `visualize_batch_in_napari`), como visor interactivo *opcional* al correr ese
script suelto para explorar el dataset crudo — con fallback automático si no hay GUI.

El script que sí generó esta carpeta evita a propósito importar napari a nivel de
módulo (ver el comentario en la cabecera del script) para no forzar esa dependencia.
Las imágenes de esta carpeta se generaron con matplotlib en backend `Agg` (sin GUI) y
se guardaron directo a disco.

## Si quieres abrir el notebook igual (sin re-entrenar)

El notebook está armado para Kaggle, con `%%writefile` de los scripts y una celda que
lanza el entrenamiento (celdas 10–11: `EPOCHS = 15`, `!{" ".join(cmd)}`). Para
inspeccionarlo sin pagar esas ~8000 s/época de nuevo:

1. Activa el entorno conda del curso: `conda env create -f ../environment.yml` (una
   vez) y luego `conda activate dlba`. Solo necesario para abrir/ejecutar celdas del
   notebook, no para ver los PNG/JSON de arriba.
2. Abre `notebook1445a8ba31.ipynb` en Jupyter/VS Code.
3. **No ejecutes las celdas 10 y 11** (son las que entrenan). Ve directo a las celdas
   14–15 ("Revisar resultados"): leen `metrics_summary.json` y muestran los PNG — pero
   asumen las rutas de Kaggle (`/kaggle/working/outputs_modulo2_pannuke`), así que si
   las corres localmente cambia `OUTPUT_DIR` a esta carpeta (`Resultados-Modulo2`).
4. La celda 16 (grilla de 8 predicciones cualitativas) sí necesita `best_model.pt`
   cargado, que **no está incluido en esta carpeta** — esa celda no se puede reproducir
   sin volver a entrenar o sin tener ese checkpoint a mano.
