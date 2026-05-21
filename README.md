# S2DR3 Black‑Box Evaluation

**No‑reference, black‑box quality assessment** of the S2DR3 super‑resolution model for Sentinel‑2.  
The model is treated as a black box – no access to architecture, weights, or training data – and evaluated solely through its outputs.

## Key idea
- **Wald protocol**: downscale the super‑resolved image to the original 10 m resolution and compare it with the real Sentinel‑2 L2A reference.
- **Quantitative metrics**:
  - Spectral: RMSE, bias, SAM (Spectral Angle Mapper), ERGAS, per‑band and overall correlation (*r*).
  - Spatial: GLCM contrast, edge density (Canny).
  - Frequency: radial power spectrum, spectral ratio, difference plots.
- **Visualisation**: bias maps (absolute and relative), SAM map, per‑band bar charts, spectral profiles for selected land‑cover classes.
- **Classification** (optional, not detailed in this update): Random Forest with manual ground truth (OA, F1, IoU).

## Why black‑box?
- Simulates a real‑world scenario where only the API or output files are available.
- Guarantees an independent, reproducible evaluation without internal knowledge of the algorithm.
- The methodology can be applied to any super‑resolution service.

## Repository structure

s2dr4-blackbox-evaluation/  
├── main.py # Main entry point  
├── src/ # Reusable modules  
│ ├── preprocess.py # Load rasters, downscale (Wald protocol)  
│ ├── spectral_metrics.py # RMSE, Bias, SAM (pixel‑wise + maps)  
│ ├── spatial_metrics.py # GLCM contrast, edge density  
│ ├── freq_metrics.py # FFT, radial power spectrum  
│ ├── classification.py # Random Forest, OA, F1, IoU  
│ └── utils.py # Visualisation helpers, config  
├── data/ # (ignored) – original S2, S2DR4 outputs, ground truth  
├── results/ # (auto‑created) – tables, plots, maps  
├── config.yaml # Configuration file containing paths, etc.  
├── requirements.txt  
├── LICENSE  
└── .gitignore  

## All implemented no‑reference metrics

| Category    | Metric                                | Module            |
|-------------|---------------------------------------|-------------------|
| Spectral    | RMSE (total + per band)               | `spectral_metrics` |
| Spectral    | SAM (mean + map)                      | `spectral_metrics` |
| Spectral    | Bias (mean radiometric shift + maps)  | `spectral_metrics` |
| Spectral    | ERGAS                                 | `spectral_metrics` |
| Spectral    | Per‑band & overall Pearson correlation | `spectral_metrics` |
| Spatial     | GLCM contrast                         | `spatial_metrics`  |
| Spatial     | Edge density (Canny)                  | `spatial_metrics`  |
| Frequency   | Radial power spectrum, slopes         | `freq_metrics`     |
| Visual      | Spectral profiles per land‑cover class | `utils`            |
| Classification | Overall Accuracy, F1, IoU           | `classification`   |  

All spectral and spatial metrics are computed **after downscaling the S2DR3 output to 10 m** (Wald protocol).

## Usage
1. **Clone repository**  
   ```bash
   git clone https://github.com/yourusername/s2dr3-blackbox-evaluation.git
   cd s2dr3-blackbox-evaluation
   ```
2. Install dependencies
    ```bash
   pip install -r requirements.txt
   ```
3. **Prepare data** (see `data/` folder structure)  
   - Place original Sentinel‑2 L2A crops (GeoTIFF, 10 m) into `data/s2_10m/`  
   - Place S2DR3 outputs (1 m) into `data/s2sr_1m/`  
   - (Optional) Place ground‑truth shapefiles for classification into `data/ground_truth/`

4. **Run the evaluation**  
   Open `main.ipynb` in Jupyter / VS Code / Colab and execute cells.  
   The notebook will:
   - Load each polygon
   - Downscale S2DR3 to 10 m
   - Compute all metrics
   - Run Random Forest classification (if ground truth available)
   - Save results to `results/`

## Data sources

- **Sentinel‑2 L2A** – [Copernicus Open Access Hub](https://scihub.copernicus.eu/)  
- **S2DR3** – public Colab notebook by Gamma Earth  
  ([Medium article](https://medium.com/@ya_71389/c71a601a2253))

## Results

The evaluation was performed on a **Sentinel‑2 L2A chip (tile T57UWA, 2024‑08‑04, ~4×4 km²)** with the S2DR3 super‑resolved output. Center coordinates: 54.4867°N, 160.0089°E (Uzon Caldera, Kamchatka).  
All metrics follow the Wald protocol – the SR image is downscaled back to 10 m and compared with the original Sentinel‑2 reference.

### Spectral accuracy

| Metric | Value | Note |
|--------|-------|------|
| Mean SAM | **2.46°** | Far below the typical 3° quality threshold |
| Total RMSE | **0.0137** | Reflectance units (0–1) |
| RMSE (native 10 m bands: B02,B03,B04,B08) | 0.0024–0.0062 | Near‑perfect agreement |
| RMSE (reconstructed 20 m bands: B05–B12) | 0.0074–0.0245 | Higher error, but still low |
| Bias (mean per band) | ±0.005 | Negligible systematic shift |
| ERGAS | 7.35 | Integrated index (lower is better) |
| Overall Pearson *r* | **0.9955** | Almost perfect linear correlation |

### Spatial quality (GLCM contrast & edge density)

GLCM contrast and edge density were computed on raw reflectance values with consistent quantisation (64 grey levels) and Otsu‑based Sobel edges – **no per‑band min‑max stretching**.

| Band type | GLCM contrast | Edge density | Interpretation |
|-----------|----------------|--------------|----------------|
| Native 10 m (B02,B03,B04,B08) | Original ≈ SR | Original ≈ SR | Texture is preserved, no over‑sharpening |
| Reconstructed 20 m (B05,B06,B07,B8A,B11,B12) | SR **2–3× higher** | SR **lower** than original | Added real fine detail; removed upsampling artifacts |

*Example:* B07 (SWIR) GLCM contrast: original 3.21 → SR 8.28; edge density: original 0.214 → SR 0.172.

### Frequency analysis (radial power spectra)

- **Native 10 m bands**: spectral slopes remain unchanged (∼ −2.7), confirming that no artificial high‑frequency noise was introduced.  
- **Reconstructed 20 m bands**: slopes become significantly flatter (from −3.5…−3.6 to −2.7…−2.9), and the ratio of SR/original power exceeds 1 at high frequencies – direct evidence of successful detail recovery without altering the overall image structure.

All figures (per‑band spectra, ratio plots, bar charts, SAM map, bias maps) are saved in `results/`.  

## Citation 

If you use this module in your work, please cite the original author:  
> Yosef Akhtman, S2DR3: Effective 12-Band 10x Single Image Super-Resolution for Sentinel-2, October 2, 2023. https://medium.com/@yakhtman/s2dr3


## License

Code in this repository is released under the **Apache 2.0**.  
Data (Sentinel‑2, S2DR3 outputs, ground truth) are subject to their own licenses.