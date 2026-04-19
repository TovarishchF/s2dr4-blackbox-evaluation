# S2DR4 Black‑Box Evaluation

**No‑reference, black‑box quality assessment** of S2DR4 super‑resolution for Sentinel‑2.  
We treat the model as a black box – no access to architecture, weights, or training data – and evaluate it solely through its outputs.

## Key idea
- Use **Wald protocol** (downscale SR → compare to original Sentinel‑2 at 10 m)
- Quantitative metrics: RMSE, SAM (spectral angle), bias maps, GLCM contrast, edge density, radial FFT spectrum
- Practical validation: Random Forest classification with manual ground‑truth (OA, F1, IoU)

## Why black‑box?
- Simulates real‑world scenario where only API/output is available
- Ensures independent, reproducible evaluation without internal knowledge
- Methodology can be applied to any super‑resolution service

## Repository structure

s2dr4-blackbox-evaluation/  
├── main.ipynb # Main entry point  
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

## All no-reference metrics implemented

| Category | Metric | Module |
|----------|--------|--------|
| Consistency | RMSE | `spectral_metrics` |
| Consistency | SAM (Spectral Angle Mapper) | `spectral_metrics` |
| Spectral distortion | Bias (mean radiometric shift + maps) | `spectral_metrics` |
| Texture | GLCM contrast | `spatial_metrics` |
| Texture | Edge density | `spatial_metrics` |
| Frequency | Radial power spectrum (FFT) | `freq_metrics` |
| Classification | Overall Accuracy (OA), F1, IoU | `classification` |

All metrics are computed after **downscaling S2DR4 output back to 10 m** (Wald protocol).  

## Usage
1. **Clone repository**  
   ```bash
   git clone https://github.com/yourusername/s2dr4-blackbox-evaluation.git
   cd s2dr4-blackbox-evaluation
   ```
2. Install dependencies
    ```bash
   pip install -r requirements.txt
   ```
3. **Prepare data** (see `data/` folder structure)  
   - Place original Sentinel‑2 L2A crops (GeoTIFF, 10 m) into `data/original_s2/`  
   - Place S2DR4 outputs (1 m) into `data/s2dr4_output/`  
   - (Optional) Place ground‑truth shapefiles for classification into `data/ground_truth/`

4. **Run the evaluation**  
   Open `main.ipynb` in Jupyter / VS Code / Colab and execute cells.  
   The notebook will:
   - Load each polygon
   - Downscale S2DR4 to 10 m
   - Compute all metrics
   - Run Random Forest classification (if ground truth available)
   - Save results to `results/`

## Data sources

- **Sentinel‑2 L2A** – [Copernicus Open Access Hub](https://scihub.copernicus.eu/)  
- **S2DR4** – public Colab notebook by Gamma Earth  
  ([Medium article](https://medium.com/@ya_71389/c71a601a2253))

## Citation (S2DR4)

If you use S2DR4 in your work, please cite:

> Yosef Akhtman, S2DR4: Effective 10‑Band 10x Single Image Super‑Resolution for Sentinel‑2. Medium, 2026.

## License

Code in this repository is released under the **MIT License**.  
Data (Sentinel‑2, S2DR4 outputs, ground truth) are subject to their own licenses.