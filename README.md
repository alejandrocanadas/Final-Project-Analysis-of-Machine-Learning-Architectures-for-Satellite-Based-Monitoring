# 🛰️ Comparative Analysis of Machine Learning Architectures for Satellite-Based Monitoring of the Langjökull–Hvítárvatn Glacial System

Project for the course **Emerging Digital Technologies 4** (Pontificia Universidad Javeriana) applying **GeoAI (Geospatial Artificial Intelligence)** techniques to automatically classify land cover of the Langjökull–Hvítárvatn glacial system (Iceland) from Sentinel-2 satellite imagery, comparing five machine learning architectures.

> Group project developed by Jerónimo Vanegas Jiménez, **Alejandro Cañadas**, Steven Bohorquez, and Juan Bernardo Uribe, under the supervision of professor Andrés Oswaldo Calderón Romero, Ph.D.

📄 The full academic report is available at [`Proyecto_Final_EDT_4_Okjökul.pdf`](./Proyecto_Final_EDT_4_Okjökul.pdf).

## 🎯 Objective

To perform supervised classification of four land cover classes (water, tundra vegetation, snow/ice, and volcanic soil/rock) over a Sentinel-2 scene of the Langjökull–Hvítárvatn system, compare the performance of five classification algorithms, and generate a predictive thematic map with area quantification per class.

## 🌍 Case Study

- **Region:** Langjökull–Hvítárvatn glacial system, west-central Iceland (Langjökull is Iceland's second-largest ice cap; Hvítárvatn is its main proglacial lake).
- **Sensor:** Sentinel-2 MSI, Level-2A product (surface reflectance, atmospherically corrected), chosen over Landsat 8 for its higher spatial resolution (10 m vs. 30 m) and richer spectral information (10 usable bands vs. 8 required).
- **Scene:** tile `T27WWM`, acquired September 28, 2025 (end of boreal summer, to minimize seasonal snow and cloud cover).
- **Bands used:** B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (visible, red edge, NIR, and SWIR).

## 🧩 Land Cover Classes

| ID | Class | Description |
|---|---|---|
| 1 | Water (Lake) | Hvítárvatn proglacial lake and meltwater streams |
| 2 | Vegetation (Tundra) | Mosses, lichens, and low-lying vegetation |
| 3 | Snow / Ice | Langjökull ice cap, bare ice, and firn |
| 4 | Soil (Volcanic rock) | Basalt, moraines, ash, and bare soil |

Four classes (rather than five) were defined because volcanic rock and bare soil showed spectral signatures too similar to separate reliably — a design decision later validated by the classification metrics.

## 🔬 Methodology (full pipeline)

### Phase 1 — Problem Definition and Data Acquisition
- Delimitation of the Region of Interest (ROI) in QGIS, stored as a GeoPackage in EPSG:32627 (WGS 84 / UTM zone 27N).
- Visual identification of the 4 classes using true-color (B4-B3-B2) and false-color infrared (B8-B4-B3) composites.
- Generation of **10,000 training points** (2,500 per class) via random sampling within manually digitized guide polygons in QGIS.

### Phase 2 — Preprocessing and Training (Python, Google Colab)
Pipeline implemented with **rasterio**, **geopandas**, and **scikit-learn**:
1. **Clipping and stacking** of the 10 bands over the ROI, resampling 20 m bands to the common 10 m grid (nearest-neighbor, no spectral value mixing) → 10-band GeoTIFF (`stack_langjokull_10m.tif`, 1547×1479 px).
2. **Pixel extraction** at the 10,000 points and export to TSV, with **per-pixel deduplication** to avoid train/test information leakage (only 3 of 10,000 points were redundant → final dataset of **9,997 unique samples**).
3. **Baseline training** (default hyperparameters) of five architectures on a stratified 70/30 split (`random_state=42`):
   - Decision Tree (DT)
   - Support Vector Machines (SVM, RBF kernel)
   - Artificial Neural Networks (ANN, MLPClassifier)
   - K-Nearest Neighbors (KNN, k=5)
   - Naive Bayes (NB)

   Scale-sensitive models (SVM, KNN, ANN) were wrapped in a pipeline with `StandardScaler`.

### Phase 3 — Optimization, Prediction, and Inference (implemented in [`fase3_langjokull.ipynb`](./fase3_langjokull.ipynb))
1. **Hyperparameter optimization** with `GridSearchCV` and 3-fold stratified cross-validation for all 5 architectures.
2. **Final evaluation** on the test set using overall accuracy, Cohen's Kappa, precision, recall, and macro F1-score.
3. **Full-scene inference**: the best model classifies every pixel of the complete raster stack.
4. **Categorical GeoTIFF generation** with the predictive thematic map, preserving the CRS and affine transform of the original scene.
5. **Area quantification** per class (hectares and km²), exported to CSV, with visualization of the thematic map and a bar chart.

## 📊 Results (Baseline, Phase 2)

Metrics on the test set (3,000 samples), default configuration (unoptimized):

| Model | Overall Accuracy | Kappa | F1-score (macro) |
|---|---|---|---|
| **ANN (neural network)** | **0.9993** | **0.9991** | **0.9993** |
| DT (decision tree) | 0.9983 | 0.9978 | 0.9983 |
| KNN (k-nearest neighbors) | 0.9973 | 0.9964 | 0.9973 |
| SVM (support vector machine) | 0.9950 | 0.9933 | 0.9950 |
| NB (Naive Bayes) | 0.9917 | 0.9889 | 0.9917 |

All 5 architectures exceeded 99% overall accuracy, confirming that the 4 defined classes are highly separable spectrally using the 10 Sentinel-2 bands. The only appreciable confusion occurred between Vegetation and Soil (between 2 and 19 pixels depending on the model), consistent with the expected spectral similarity between sparse tundra and volcanic rock/soil.

## 🛠️ Technologies Used

| Category | Tool |
|---|---|
| Language | Python (Google Colab) |
| Raster/vector processing | rasterio, geopandas |
| Machine Learning | scikit-learn (DecisionTree, SVM, MLPClassifier, KNN, GaussianNB, GridSearchCV) |
| Visualization | matplotlib |
| GIS / spatial delimitation | QGIS |
| Satellite data source | Copernicus Data Space (Sentinel-2 MSI Level-2A) |

## 📁 Repository Structure

```
Final-Project-Analysis-of-Machine-Learning-Architectures/
├── Proyecto_Final_EDT_4_Okjökul.pdf     # Full academic report (Phases 1 and 2)
├── fase2_langjokull_colab_3(1).ipynb     # Preprocessing and baseline training
├── fase3_langjokull.ipynb                 # HPO, final evaluation, inference, and area quantification
└── README.md
```

## ⚙️ How to Run the Project

The notebooks are designed to run in **Google Colab** with Google Drive mounted (input/output paths point to Drive folders).

1. Upload the Sentinel-2 scene downloaded from [Copernicus Data Space](https://dataspace.copernicus.eu/) and the training points GeoPackage to your Google Drive.
2. Open `fase2_langjokull_colab_3(1).ipynb` in Colab, adjust the input paths, and run it to generate the raster stack and training TSV.
3. Open `fase3_langjokull.ipynb`, adjust `STACK_PATH` and `TSV_PATH` according to the Phase 2 outputs, and run it to obtain:
   - Final comparative metrics table (`metrics_final_fase3.csv`)
   - Classified thematic map (`mapa_clasificado_langjokull.tif`)
   - Area quantification per class (`areas_por_clase.csv`)
4. (Optional) Import the resulting GeoTIFF into QGIS to visualize it with unique-values symbology per class.

## 📌 Key Conclusions

- Sentinel-2 Level-2A, with its 10 spectral bands at 10-20 m resolution, enables highly precise discrimination (>99% OA) of the 4 land cover types in the studied glacial-lacustrine system.
- The neural network (ANN) achieved the best baseline performance, closely followed by the decision tree and KNN.
- Per-pixel deduplication confirmed that the high performance reflects genuine spectral separability, not train/test information leakage.
- As an acknowledged limitation, sampling from contiguous polygons introduces some spatial autocorrelation; a spatially disjoint validation would be the next step toward an even more rigorous comparison between architectures.

## 👤 Authors

- Jerónimo Vanegas Jiménez
- [Alejandro Cañadas](https://github.com/alejandrocanadas)
