import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import os
import pyproj
import rasterio
import matplotlib.pyplot as plt

os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (load_config, setup_logging, save_colormap_image,
                       plot_rmse_bars, plot_bias_bars, plot_per_band_bars, plot_spectral_profiles)
from src.preprocess import load_data, downsample_sr_to_original, crop_to_common_extent
from src.spectral_metrics import rmse, bias, sam, channel_rmse, ergas, channel_correlation, overall_correlation
from src.spatial_metrics import glcm_contrast_multiband, edge_density_multiband
from src.freq_metrics import radial_spectrum, plot_radial_spectrum, plot_radial_spectrum_ratio, plot_spectrum_diff

def main():
    start_time = datetime.now()
    logger = setup_logging(log_file="logs/eval.log", level=logging.INFO)
    logger.info(f"=== Начало оценки S2DR3 — {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    config = load_config("config.yaml")
    logger.info("Конфигурация загружена")

    plt.rcParams.update({
        'font.family': 'arial',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

    orig_stack, sr_stack, orig_geoinfo, sr_geoinfo = load_data(config)
    logger.info(f"Исходный S2: форма {orig_stack.shape}, dtype={orig_stack.dtype}")
    logger.info(f"SR: форма {sr_stack.shape}, dtype={sr_stack.dtype}")

    orig_cropped, sr_cropped, geoinfo = crop_to_common_extent(
        orig_stack, orig_geoinfo, sr_stack, sr_geoinfo
    )
    logger.info(f"После обрезки — исходный: {orig_cropped.shape}")
    logger.info(f"После обрезки — SR: {sr_cropped.shape}")

    target_shape = orig_cropped.shape[1:]
    downsampled = downsample_sr_to_original(sr_cropped, target_shape)
    logger.info(f"Пониженный до разрешения SR: форма {downsampled.shape}")

    orig_cropped /= 10000
    downsampled /= 10000

    total_rmse = rmse(orig_cropped, downsampled)
    logger.info(f"Общий RMSE (ед. отражения): {total_rmse:.6f}")

    channel_rmse_vals = channel_rmse(orig_cropped, downsampled)
    for band, val in zip(config['bands'], channel_rmse_vals):
        logger.info(f"RMSE {band}: {val:.6f}")

    corr_per_band = channel_correlation(orig_cropped, downsampled)
    corr_overall = overall_correlation(orig_cropped, downsampled)
    for band, r in zip(config['bands'], corr_per_band):
        logger.info(f"Корреляция (r) {band}: {r:.4f}")
    logger.info(f"Общая корреляция: {corr_overall:.4f}")

    mean_bias, bias_map = bias(orig_cropped, downsampled)
    for band, val in zip(config['bands'], mean_bias):
        band_index = config['bands'].index(band)
        save_colormap_image(
            bias_map[band_index],
            f"results/maps/Bias/bias_{band}.png",
            geoinfo['transform'],
            geoinfo['crs'],
            cmap='coolwarm',
            title=f'Смещение {band}',
            draw_grid=True,
            bounds=geoinfo['bounds']
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            pbias = (downsampled[band_index] - orig_cropped[band_index]) / orig_cropped[band_index] * 100.0
        pbias[orig_cropped[band_index] < 1e-6] = np.nan

        max_abs = max(abs(np.nanmin(pbias)), abs(np.nanmax(pbias)))
        symmetric_vmax = 20

        save_colormap_image(
            pbias,
            f"results/maps/Bias/bias_percent_{band}.png",
            geoinfo['transform'],
            geoinfo['crs'],
            cmap='coolwarm',
            title=f'Смещение {band} (%)',
            draw_grid=True,
            bounds=geoinfo['bounds'],
            vmin=-symmetric_vmax,
            vmax=symmetric_vmax
        )

        logger.info(f"Смещение {band}: {val:.6f}")

    mean_sam, sam_map = sam(orig_cropped, downsampled)
    save_colormap_image(
        sam_map,
        "results/maps/sam_mean.png",
        geoinfo['transform'],
        geoinfo['crs'],
        cmap='viridis',
        title='SAM (град.)',
        draw_grid=True,
        bounds=geoinfo['bounds']
    )
    logger.info(f"Средний SAM: {mean_sam:.2f}°")

    ergas_value = ergas(orig_cropped, downsampled)
    logger.info(f"Общий ERGAS: {ergas_value:.4f}")

    glcm_orig = glcm_contrast_multiband(orig_cropped, levels=64)
    glcm_down = glcm_contrast_multiband(downsampled, levels=64)
    ed_orig = edge_density_multiband(orig_cropped, method='sobel')
    ed_down = edge_density_multiband(downsampled, method='sobel')

    for i, band in enumerate(config['bands']):
        logger.info(f"Контраст GLCM (исходный) {band}: {glcm_orig[i]:.4f}")
        logger.info(f"Контраст GLCM (SR) {band}: {glcm_down[i]:.4f}")
        logger.info(f"Плотность границ (исходный) {band}: {ed_orig[i]:.4f}")
        logger.info(f"Плотность границ (SR) {band}: {ed_down[i]:.4f}")

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    plot_rmse_bars(channel_rmse_vals, config['bands'],
                   'СКО (отражение)', 'Поканальная среднеквадратическая ошибка',
                   'results/figures/rmse_bars.png')

    plot_bias_bars(mean_bias, config['bands'], 'results/figures/bias_bars.png')

    plot_per_band_bars(glcm_orig, glcm_down, config['bands'],
                       'Контраст GLCM', 'Поканальный контраст GLCM',
                       'results/figures/glcm_bars.png')
    plot_per_band_bars(ed_orig, ed_down, config['bands'],
                       'Плотность границ', 'Поканальная плотность границ',
                       'results/figures/edge_density_bars.png')

    Path("results/fft").mkdir(parents=True, exist_ok=True)
    slopes_orig = []
    slopes_sr = []

    all_freqs = []
    all_pow_orig = []
    all_pow_sr = []
    for i, band in enumerate(config['bands']):
        freqs, pow_orig = radial_spectrum(orig_cropped[i], pixel_size=10.0)
        _, pow_sr = radial_spectrum(downsampled[i], pixel_size=10.0)
        all_freqs.append(freqs)
        all_pow_orig.append(pow_orig)
        all_pow_sr.append(pow_sr)

    freq_lims = [0.001, 0.05]
    common_freqs = all_freqs[0]
    mask = (common_freqs >= freq_lims[0]) & (common_freqs <= freq_lims[1])
    all_pow_vals = np.concatenate([p[mask] for p in all_pow_orig + all_pow_sr])
    pow_min = np.min(all_pow_vals[all_pow_vals > 0]) * 0.5
    pow_max = np.max(all_pow_vals) * 2.0
    pow_lims = [pow_min, pow_max]

    ratio_lims = [0.2, 10e1]
    diff_lims = [-0.2, 1.3]

    for i, band in enumerate(config['bands']):
        freqs = all_freqs[i]
        power_orig = all_pow_orig[i]
        power_sr = all_pow_sr[i]

        plot_radial_spectrum(freqs, power_orig, power_sr,
                             save_path=f"results/fft/radial_spectrum_{band}.png",
                             title=f"{band}",
                             xlim=freq_lims, ylim=pow_lims)
        plot_radial_spectrum_ratio(freqs, power_orig, power_sr,
                                   save_path=f"results/fft/ratio_{band}.png",
                                   title=band,
                                   xlim=freq_lims, ylim=ratio_lims)
        plot_spectrum_diff(freqs, power_orig, power_sr,
                           save_path=f"results/fft/difference_{band}.png",
                           xlim=freq_lims, ylim=diff_lims)

        valid = (power_orig > 0) & (power_sr > 0)
        if np.sum(valid) > 3:
            log_freq = np.log10(freqs[valid])
            log_power_orig = np.log10(power_orig[valid])
            log_power_sr = np.log10(power_sr[valid])
            slope_orig = np.polyfit(log_freq, log_power_orig, 1)[0]
            slope_sr = np.polyfit(log_freq, log_power_sr, 1)[0]
        else:
            slope_orig = slope_sr = np.nan
        slopes_orig.append(slope_orig)
        slopes_sr.append(slope_sr)
        logger.info(f"Наклон радиального спектра {band}: исходный={slope_orig:.3f}, SR={slope_sr:.3f}")

    import pandas as pd
    metrics_dict = {
        'total_rmse': total_rmse,
        'mean_sam_deg': mean_sam,
        'glcm_contrast_orig': glcm_orig,
        'glcm_contrast_sr': glcm_down,
        'edge_density_orig': ed_orig,
        'edge_density_sr': ed_down,
    }
    for i, band in enumerate(config['bands']):
        metrics_dict[f'rmse_{band}'] = channel_rmse_vals[i]
        metrics_dict[f'bias_{band}'] = mean_bias[i]
        metrics_dict[f'glcm_orig_{band}'] = glcm_orig[i]
        metrics_dict[f'glcm_sr_{band}'] = glcm_down[i]
        metrics_dict[f'ed_orig_{band}'] = ed_orig[i]
        metrics_dict[f'ed_sr_{band}'] = ed_down[i]
        metrics_dict[f'slope_orig_{band}'] = slopes_orig[i]
        metrics_dict[f'slope_sr_{band}'] = slopes_sr[i]
    df = pd.DataFrame([metrics_dict])
    df.to_csv("results/metrics_summary.csv", index=False)

    classes = {
        'Вода': [(144, 51)],
        'Трава': [(249, 119)],
        'Лес': [(334, 364)],
        'Почва': [(256, 153)],
    }

    plot_spectral_profiles(orig_cropped, downsampled, classes,
                            config['bands'], 'results/figures',
                            window=3)

    end_time = datetime.now()
    logger.info("Метрики сохранены в CSV")
    logger.info(f"=== Оценка завершена — {end_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    logger.info(f"Общее время выполнения: {end_time - start_time}")


if __name__ == "__main__":
    main()