import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import sys
import os
import pyproj

os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (load_config, setup_logging, save_colormap_image,
                       plot_rmse_bars, plot_bias_bars, plot_per_band_bars, ensure_territory_dirs)
from src.preprocess import load_data, downsample_sr_to_original, crop_to_common_extent
from src.spectral_metrics import rmse, bias, sam, channel_rmse, ergas, channel_correlation, overall_correlation
from src.spatial_metrics import glcm_contrast_multiband, edge_density_multiband
from src.freq_metrics import radial_spectrum, plot_radial_spectrum, plot_spectrum_diff
from src.locale import set_language, get

def process_territory(territory_cfg, bands, results_root, logger):
    name = territory_cfg['name']
    logger.info(get('log_territory', name))

    out_base = ensure_territory_dirs(name, results_root)
    fig_dir = out_base / "figures"
    fft_dir = out_base / "fft"
    maps_dir = out_base / "maps" / "Bias"
    csv_dir = out_base

    orig_stack, sr_stack, orig_geoinfo, sr_geoinfo = load_data(territory_cfg)
    logger.info(get('log_shape_orig', orig_stack.shape, orig_stack.dtype))
    logger.info(get('log_shape_sr', sr_stack.shape, sr_stack.dtype))

    orig_cropped, sr_cropped, geoinfo = crop_to_common_extent(
        orig_stack, orig_geoinfo, sr_stack, sr_geoinfo
    )
    logger.info(get('log_cropped_orig', orig_cropped.shape))
    logger.info(get('log_cropped_sr', sr_cropped.shape))

    target_shape = orig_cropped.shape[1:]
    downsampled = downsample_sr_to_original(sr_cropped, target_shape)
    logger.info(get('log_downsampled', downsampled.shape))

    orig_cropped /= 10000.0
    downsampled /= 10000.0

    total_rmse = rmse(orig_cropped, downsampled)
    logger.info(get('log_rmse_total', total_rmse))

    ch_rmse = channel_rmse(orig_cropped, downsampled)
    for b, val in zip(bands, ch_rmse):
        logger.info(get('log_rmse_band', b, val))

    corr_per_band = channel_correlation(orig_cropped, downsampled)
    corr_overall = overall_correlation(orig_cropped, downsampled)
    for b, r in zip(bands, corr_per_band):
        logger.info(get('log_corr_band', b, r))
    logger.info(get('log_corr_overall', corr_overall))

    mean_bias, bias_map = bias(orig_cropped, downsampled)
    for idx, b in enumerate(bands):
        save_colormap_image(
            bias_map[idx],
            str(maps_dir / f"bias_{b}.png"),
            geoinfo['transform'],
            geoinfo['crs'],
            cmap='coolwarm',
            title=get('bias_map_title', b),
            draw_grid=True,
            bounds=geoinfo['bounds']
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            pbias = (downsampled[idx] - orig_cropped[idx]) / orig_cropped[idx] * 100.0
        pbias[orig_cropped[idx] < 1e-6] = np.nan
        symmetric_vmax = 60
        save_colormap_image(
            pbias,
            str(maps_dir / f"bias_percent_{b}.png"),
            geoinfo['transform'],
            geoinfo['crs'],
            cmap='coolwarm',
            title=get('bias_percent_title', b),
            draw_grid=True,
            bounds=geoinfo['bounds'],
            vmin=-symmetric_vmax,
            vmax=symmetric_vmax
        )
        logger.info(get('log_bias_band', b, mean_bias[idx]))

    mean_sam, sam_map = sam(orig_cropped, downsampled)
    save_colormap_image(
        sam_map,
        str(out_base / "maps" / "sam_mean.png"),
        geoinfo['transform'],
        geoinfo['crs'],
        cmap='viridis',
        title=get('sam_map_title'),
        draw_grid=True,
        bounds=geoinfo['bounds']
    )
    logger.info(get('log_sam', mean_sam))

    ergas_val = ergas(orig_cropped, downsampled)
    logger.info(get('log_ergas', ergas_val))

    glcm_orig = glcm_contrast_multiband(orig_cropped, levels=64)
    glcm_down = glcm_contrast_multiband(downsampled, levels=64)
    ed_orig = edge_density_multiband(orig_cropped, method='canny')
    ed_down = edge_density_multiband(downsampled, method='canny')

    for i, b in enumerate(bands):
        logger.info(get('log_glcm', b, glcm_orig[i], glcm_down[i]))
        logger.info(get('log_edge', b, ed_orig[i], ed_down[i]))

    plot_rmse_bars(ch_rmse, bands,
                   get('ylabel_rmse'), get('plot_title_rmse'),
                   str(fig_dir / 'rmse_bars.png'))
    plot_bias_bars(mean_bias, bands, str(fig_dir / 'bias_bars.png'))
    plot_per_band_bars(glcm_orig, glcm_down, bands,
                       get('ylabel_glcm'), get('plot_title_glcm'),
                       str(fig_dir / 'glcm_bars.png'))
    plot_per_band_bars(ed_orig, ed_down, bands,
                       get('ylabel_edge'), get('plot_title_edge'),
                       str(fig_dir / 'edge_density_bars.png'))

    slopes_orig = []
    slopes_sr = []
    all_freqs = []
    all_pow_orig = []
    all_pow_sr = []
    for i, b in enumerate(bands):
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
    diff_lims = [-0.2, 1.3]

    for i, b in enumerate(bands):
        freqs = all_freqs[i]
        power_orig = all_pow_orig[i]
        power_sr = all_pow_sr[i]
        plot_radial_spectrum(freqs, power_orig, power_sr,
                             save_path=str(fft_dir / f'radial_spectrum_{b}.png'),
                             title=get('plot_title_radial'),
                             xlim=freq_lims, ylim=pow_lims)
        plot_spectrum_diff(freqs, power_orig, power_sr,
                           save_path=str(fft_dir / f'difference_{b}.png'),
                           title=get('plot_title_diff'),
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
        logger.info(get('log_slope', b, slope_orig, slope_sr))

    metrics_dict = {
        'territory': name,
        'total_rmse': total_rmse,
        'mean_sam_deg': mean_sam,
        'ergas': ergas_val,
        'overall_corr': corr_overall,
    }
    for i, b in enumerate(bands):
        metrics_dict[f'rmse_{b}'] = ch_rmse[i]
        metrics_dict[f'bias_{b}'] = mean_bias[i]
        metrics_dict[f'glcm_orig_{b}'] = glcm_orig[i]
        metrics_dict[f'glcm_sr_{b}'] = glcm_down[i]
        metrics_dict[f'ed_orig_{b}'] = ed_orig[i]
        metrics_dict[f'ed_sr_{b}'] = ed_down[i]
        metrics_dict[f'slope_orig_{b}'] = slopes_orig[i]
        metrics_dict[f'slope_sr_{b}'] = slopes_sr[i]

    df = pd.DataFrame([metrics_dict])
    csv_path = csv_dir / f"{name}_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(get('log_saved_csv', csv_path))

    logger.info(get('log_territory_done', name))
    return metrics_dict

def main():
    start_time = datetime.now()
    config = load_config("config.yaml")
    set_language(config.get('language', 'ru'))
    logger = setup_logging(log_file="logs/eval.log", level=config['logging']['level'])
    logger.info(get('log_start'))

    territories = config['territories']
    bands = config['bands']
    results_root = config['results_root']

    all_metrics = []
    for t in territories:
        try:
            metrics = process_territory(t, bands, results_root, logger)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(get('log_error', t.get('name', 'unknown'), str(e)), exc_info=True)

    if all_metrics:
        df_all = pd.DataFrame(all_metrics)
        df_all.to_csv(Path(results_root) / "all_territories_summary.csv", index=False)
        logger.info(get('log_aggregated'))
    else:
        logger.warning(get('log_no_territories'))

    end_time = datetime.now()
    logger.info(get('log_end', end_time.strftime('%Y-%m-%d %H:%M:%S'), str(end_time - start_time)))

if __name__ == "__main__":
    main()