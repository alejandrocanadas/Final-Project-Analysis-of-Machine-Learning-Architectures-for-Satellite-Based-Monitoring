# 🛰️ Análisis Comparativo de Arquitecturas de Machine Learning para el Monitoreo Satelital del Sistema Glaciar Langjökull–Hvítárvatn

Proyecto de la asignatura **Tecnologías Digitales Emergentes 4** (Pontificia Universidad Javeriana) que aplica técnicas de **GeoAI (Geospatial Artificial Intelligence)** para clasificar automáticamente la cobertura del suelo del sistema glaciar Langjökull–Hvítárvatn (Islandia) a partir de imágenes satelitales Sentinel-2, comparando cinco arquitecturas de machine learning.

> Proyecto grupal desarrollado por Jerónimo Vanegas Jiménez, **Alejandro Cañadas**, Steven Bohorquez y Juan Bernardo Uribe, bajo la supervisión del profesor Andrés Oswaldo Calderón Romero, Ph.D.

📄 El informe académico completo está disponible en [`Proyecto_Final_EDT_4_Okjökul.pdf`](./Proyecto_Final_EDT_4_Okjökul.pdf).

## 🎯 Objetivo

Clasificar de forma supervisada cuatro coberturas del suelo (agua, vegetación de tundra, nieve/hielo y suelo/roca volcánica) sobre una escena Sentinel-2 del sistema Langjökull–Hvítárvatn, comparar el desempeño de cinco algoritmos de clasificación, y generar un mapa temático predictivo con cuantificación de áreas por clase.

## 🌍 Caso de estudio

- **Región:** sistema glaciar Langjökull–Hvítárvatn, centro-oeste de Islandia (Langjökull es el segundo casquete glaciar más grande del país; Hvítárvatn es su principal lago proglaciar).
- **Sensor:** Sentinel-2 MSI, producto Level-2A (reflectancia superficial, corregido atmosféricamente), elegido sobre Landsat 8 por su mayor resolución espacial (10 m vs. 30 m) y mayor riqueza espectral (10 bandas útiles vs. 8 requeridas).
- **Escena:** tile `T27WWM`, adquirida el 28 de septiembre de 2025 (fin del verano boreal, para minimizar nieve estacional y nubosidad).
- **Bandas usadas:** B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (visible, red edge, NIR y SWIR).

## 🧩 Clases de cobertura

| ID | Clase | Descripción |
|---|---|---|
| 1 | Agua (Lake) | Lago proglaciar Hvítárvatn y arroyos de deshielo |
| 2 | Vegetación (Tundra) | Musgos, líquenes y vegetación rasa |
| 3 | Nieve / Hielo (Snow) | Casquete glaciar Langjökull, hielo desnudo y firn |
| 4 | Suelo (Roca volcánica) | Basalto, morrenas, ceniza y suelo desnudo |

Se definieron 4 clases (y no 5) porque roca volcánica y suelo desnudo mostraron firmas espectrales demasiado similares para separarlas de forma robusta — una decisión de diseño validada posteriormente por las métricas de clasificación.

## 🔬 Metodología (pipeline completo)

### Fase 1 — Definición del problema y adquisición de datos
- Delimitación de la Región de Interés (ROI) en QGIS, almacenada como GeoPackage en EPSG:32627 (WGS 84 / UTM zona 27N).
- Identificación visual de las 4 clases mediante composiciones en color verdadero (B4-B3-B2) y falso color infrarrojo (B8-B4-B3).
- Generación de **10,000 puntos de entrenamiento** (2,500 por clase) mediante muestreo aleatorio dentro de polígonos guía digitalizados manualmente en QGIS.

### Fase 2 — Preprocesamiento y entrenamiento (Python, Google Colab)
Pipeline implementado con **rasterio**, **geopandas** y **scikit-learn**:
1. **Recorte y apilamiento** de las 10 bandas sobre la ROI, remuestreando las bandas de 20 m a la grilla común de 10 m (vecino más cercano, sin mezclar valores espectrales) → GeoTIFF de 10 bandas (`stack_langjokull_10m.tif`, 1547×1479 px).
2. **Extracción de píxeles** en los 10,000 puntos y exportación a TSV, con **deduplicación por píxel** para evitar fuga de información entre train/test (solo 3 de 10,000 puntos resultaron redundantes → dataset final de **9,997 muestras únicas**).
3. **Entrenamiento baseline** (hiperparámetros por defecto) de cinco arquitecturas sobre partición estratificada 70/30 (`random_state=42`):
   - Árbol de decisión (DT)
   - Máquinas de soporte vectorial (SVM, kernel RBF)
   - Redes neuronales artificiales (ANN, MLPClassifier)
   - K-vecinos más cercanos (KNN, k=5)
   - Naive Bayes (NB)

   Los modelos sensibles a escala (SVM, KNN, ANN) se encapsularon en un pipeline con `StandardScaler`.

### Fase 3 — Optimización, predicción e inferencia (implementada en [`fase3_langjokull.ipynb`](./fase3_langjokull.ipynb))
1. **Optimización de hiperparámetros** con `GridSearchCV` y validación cruzada estratificada de 3 folds para las 5 arquitecturas.
2. **Evaluación final** sobre el conjunto de test con overall accuracy, Kappa de Cohen, precisión, recall y F1-score (macro).
3. **Inferencia full-scene**: el mejor modelo clasifica cada píxel del raster stack completo.
4. **Generación de GeoTIFF categórico** con el mapa temático predictivo, conservando CRS y transformación afín de la escena original.
5. **Cuantificación de áreas** por clase (hectáreas y km²), exportadas a CSV, con visualización del mapa temático y gráfico de barras.

## 📊 Resultados (línea base, Fase 2)

Métricas sobre el conjunto de prueba (3,000 muestras), configuración por defecto (sin optimizar):

| Modelo | Overall Accuracy | Kappa | F1-score (macro) |
|---|---|---|---|
| **ANN (red neuronal)** | **0.9993** | **0.9991** | **0.9993** |
| DT (árbol de decisión) | 0.9983 | 0.9978 | 0.9983 |
| KNN (k-vecinos) | 0.9973 | 0.9964 | 0.9973 |
| SVM (soporte vectorial) | 0.9950 | 0.9933 | 0.9950 |
| NB (Naive Bayes) | 0.9917 | 0.9889 | 0.9917 |

Las 5 arquitecturas superaron el 99% de overall accuracy, confirmando que las 4 clases definidas son espectralmente muy separables con las 10 bandas de Sentinel-2. La única confusión apreciable ocurrió entre Vegetación y Suelo (entre 2 y 19 píxeles según el modelo), lo cual es coherente con la similitud espectral esperada entre tundra rala y roca/suelo volcánico.

## 🛠️ Tecnologías utilizadas

| Categoría | Herramienta |
|---|---|
| Lenguaje | Python (Google Colab) |
| Procesamiento raster/vectorial | rasterio, geopandas |
| Machine Learning | scikit-learn (DecisionTree, SVM, MLPClassifier, KNN, GaussianNB, GridSearchCV) |
| Visualización | matplotlib |
| SIG / delimitación espacial | QGIS |
| Fuente de datos satelitales | Copernicus Data Space (Sentinel-2 MSI Level-2A) |

## 📁 Estructura del repositorio

```
Final-Project-Analysis-of-Machine-Learning-Architectures/
├── Proyecto_Final_EDT_4_Okjökul.pdf     # Informe académico completo (Fases 1 y 2)
├── fase2_langjokull_colab_3(1).ipynb     # Preprocesamiento y entrenamiento baseline
├── fase3_langjokull.ipynb                 # HPO, evaluación final, inferencia y cuantificación de áreas
└── README.md
```

## ⚙️ Cómo ejecutar el proyecto

Los notebooks están diseñados para correr en **Google Colab** con Google Drive montado (los paths de entrada/salida apuntan a carpetas de Drive).

1. Sube la escena Sentinel-2 descargada de [Copernicus Data Space](https://dataspace.copernicus.eu/) y el GeoPackage de puntos de entrenamiento a tu Google Drive.
2. Abre `fase2_langjokull_colab_3(1).ipynb` en Colab, ajusta las rutas de entrada, y ejecútalo para generar el raster stack y el TSV de entrenamiento.
3. Abre `fase3_langjokull.ipynb`, ajusta `STACK_PATH` y `TSV_PATH` según las salidas de la Fase 2, y ejecútalo para obtener:
   - Tabla comparativa final de métricas (`metrics_final_fase3.csv`)
   - Mapa temático clasificado (`mapa_clasificado_langjokull.tif`)
   - Cuantificación de áreas por clase (`areas_por_clase.csv`)
4. (Opcional) Importa el GeoTIFF resultante en QGIS para visualizarlo con simbología de valores únicos por clase.

## 📌 Conclusiones clave

- Sentinel-2 Level-2A, con sus 10 bandas espectrales a 10-20 m de resolución, permite discriminar con altísima precisión (>99% OA) las 4 coberturas del sistema glaciar-lacustre estudiado.
- La red neuronal (ANN) obtuvo el mejor desempeño baseline, seguida de cerca por el árbol de decisión y KNN.
- La deduplicación por píxel confirmó que el alto desempeño refleja separabilidad espectral genuina, no fuga de información entre entrenamiento y prueba.
- Como limitación reconocida, el muestreo por polígonos contiguos introduce cierta autocorrelación espacial; una validación espacialmente disjunta sería el siguiente paso para una comparación aún más rigurosa entre arquitecturas.

## 👤 Autores

- Jerónimo Vanegas Jiménez
- [Alejandro Cañadas](https://github.com/alejandrocanadas)
