# Resultados del Módulo 2 — Segmentación binaria de núcleos con U-Net (PanNuke)

## Propósito de esta carpeta

Esta carpeta contiene los resultados de una ejecución completa del pipeline de
entrenamiento y evaluación del Módulo 2, correspondiente a la tarea de
segmentación semántica binaria de núcleos celulares sobre el conjunto de datos
PanNuke. Los artefactos aquí presentes fueron generados por el script
[`modulo2_pannuke_binary_segmentation.py`](../modulo2_pannuke_binary_segmentation.py)
y constituyen evidencia directa del comportamiento del modelo entrenado; no es
necesario volver a ejecutar el entrenamiento para revisarlos o evaluarlos.

## Contenido de la carpeta

| Archivo | Descripción |
|---|---|
| `metrics_summary.json` | Resumen cuantitativo de la corrida: métricas finales de test (Dice, IoU), dispositivo de cómputo utilizado y tiempo total de entrenamiento. |
| `training_history.png` | Curva de evolución de la función de pérdida y las métricas de validación a lo largo de las épocas de entrenamiento. |
| `qualitative_predictions.png` | Panel comparativo con ejemplos representativos: imagen de entrada (H&E), máscara de referencia, mapa de probabilidad predicho y predicción binarizada. |
| `notebook1445a8ba31.ipynb` | Notebook de ejecución (preparado para el entorno Kaggle) que orquestó la corrida y produjo los artefactos anteriores. |

## Resumen de resultados

| Métrica | Valor |
|---|---|
| Dice (test) | 0.8283 |
| IoU (test) | 0.7228 |
| Épocas de entrenamiento | 15 |
| Tiempo de entrenamiento | ≈ 105.3 min (GPU) |

La comparación cuantitativa contra el modelo de referencia (`model.h5`) no pudo
realizarse en esta corrida por una incompatibilidad entre la capa
`Conv2DTranspose` del modelo `.h5` original y la versión de Keras 3 utilizada;
esta condición queda registrada en el campo `note` de `metrics_summary.json`.

## Cómo revisar los resultados

Dado que el objetivo es la revisión de resultados ya obtenidos y no la
reproducción del entrenamiento, basta con consultar directamente los archivos
estáticos incluidos en esta carpeta:

- `metrics_summary.json` puede abrirse con cualquier editor de texto.
- `training_history.png` y `qualitative_predictions.png` pueden abrirse con
  cualquier visor de imágenes.

No se requiere entorno de Python, Jupyter ni GPU para esta revisión.

## Nota sobre la dependencia `napari`

El archivo `environment.yml` del proyecto incluye `napari[all]` como
dependencia, pero esta librería **no interviene en la generación ni en la
visualización de los resultados de esta carpeta**. Su único uso dentro del
proyecto es como visor interactivo opcional del conjunto de datos crudo,
implementado en la función `visualize_batch_in_napari` de
[`05_dataset_exploration_and_dataloaders.py`](../05_dataset_exploration_and_dataloaders.py),
con reversión automática a modo no interactivo cuando no hay entorno gráfico
disponible.

El script que produjo los artefactos de esta carpeta evita deliberadamente la
importación de `napari` a nivel de módulo, precisamente para no introducir esa
dependencia en el flujo de entrenamiento y evaluación (véase el comentario
correspondiente en la cabecera del script). Las figuras se generaron con
`matplotlib` en modo no interactivo (backend `Agg`) y se guardaron
directamente a disco.

## Instrucciones opcionales para reabrir el notebook

La reproducción íntegra del notebook no es necesaria para interpretar los
resultados aquí incluidos; se documenta a continuación únicamente como
referencia, para quien desee inspeccionar el proceso de ejecución sin
reentrenar el modelo (el reentrenamiento completo insume aproximadamente
8000 segundos por época en el hardware disponible).

1. Crear y activar el entorno del curso una única vez:

   ```bash
   conda env create -f ../environment.yml
   conda activate dlba
   ```

2. Abrir `notebook1445a8ba31.ipynb` en Jupyter o en un editor compatible
   (por ejemplo, Visual Studio Code).
3. Omitir la ejecución de las celdas 10 y 11, correspondientes al lanzamiento
   del entrenamiento (`EPOCHS = 15` y la llamada al script principal).
4. Ejecutar directamente las celdas 14 y 15 ("Revisar resultados"), que leen
   `metrics_summary.json` y muestran las figuras generadas. Estas celdas
   asumen las rutas del entorno Kaggle
   (`/kaggle/working/outputs_modulo2_pannuke`); para ejecutarlas localmente
   debe redefinirse `OUTPUT_DIR` apuntando a esta carpeta
   (`Resultados-Modulo2`).
5. La celda 16, que reconstruye una grilla adicional de ocho predicciones
   cualitativas, requiere cargar el checkpoint `best_model.pt`, el cual **no
   está incluido en esta carpeta**. Dicha celda no puede reproducirse sin
   dicho archivo o sin repetir el entrenamiento.
