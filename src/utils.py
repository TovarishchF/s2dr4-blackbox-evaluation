import logging
import yaml
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

def setup_logging(log_file=None, level=logging.INFO):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    return logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def write_raster(filepath, array, transform, crs):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        filepath, 'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array, 1)

def find_stack_ms_file(directory: str) -> Path:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Директория {directory} не существует")
    ms_files = list(directory.glob("*_MS.tif"))
    if not ms_files:
        raise FileNotFoundError(f"Не найден файл *_MS.tif в {directory}")
    return ms_files[0]

def load_stack_ms(filepath: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(filepath) as src:
        stack = src.read().astype(np.float32)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    geoinfo = {'transform': transform, 'crs': crs, 'bounds': bounds}
    return stack, geoinfo

def save_colormap_image(array, output_path,
                        transform, crs, cmap='coolwarm',
                        title=None, draw_grid=False,
                        bounds=None, vmin=None, vmax=None):
    tif_path = output_path.replace('.png', '.tif')
    write_raster(tif_path, array, transform, crs)
    if draw_grid:
        try:
            import cartopy.crs as ccrs
            from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            epsg_code = crs.to_epsg()
            if epsg_code:
                src_crs = ccrs.epsg(epsg_code)
            else:
                src_crs = ccrs.PlateCarree()
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=src_crs)
            if bounds is not None:
                left, bottom, right, top = bounds
                extent = (left, right, bottom, top)
            else:
                left = transform.c
                top = transform.f
                right = left + array.shape[1] * transform.a
                bottom = top + array.shape[0] * transform.e
                extent = (left, right, bottom, top)
            im = ax.imshow(array, cmap=cmap, extent=extent, transform=src_crs, origin='upper', vmin=vmin, vmax=vmax)
            ax.set_extent(extent, crs=src_crs)
            plt.colorbar(im, ax=ax, label=title if title else 'Value')
            ax.set_title(title)
            gl = ax.gridlines(draw_labels=True, dms=True, linestyle='-', color='black', alpha=0.7)
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LongitudeFormatter(dms=True, auto_hide=False)
            gl.yformatter = LatitudeFormatter(dms=True, auto_hide=False)
            gl.xlocator = mticker.MultipleLocator(30*2/3600)
            gl.ylocator = mticker.MultipleLocator(15*2/3600)
            gl.xlabel_style = {'size': 8, 'color': 'black', 'rotation': 60}
            gl.ylabel_style = {'size': 8, 'color': 'black'}
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        except ImportError:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(array, cmap=cmap, interpolation='nearest')
            plt.colorbar(im, ax=ax, label=title if title else 'Value')
            ax.set_title(title)
            ax.axis('off')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(array, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=title if title else 'Value')
        ax.set_title(title)
        ax.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def plot_rmse_bars(vals, band_names, ylabel, title, save_path):
    x = np.arange(len(band_names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, vals, color='#b2182b', edgecolor='white', linewidth=0.5, label='S2DR3')
    ax.set_xlabel('Канал', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, fontsize=10)
    ax.legend(fontsize=10, frameon=True, edgecolor='grey', facecolor='white')
    ax.grid(True, axis='y', which='major', linestyle='-', linewidth=0.4, color='grey', alpha=0.4)
    ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    ax.grid(True, axis='y', which='minor', linestyle='-', linewidth=0.2, color='grey', alpha=0.2)
    ax.set_facecolor('#f7f7f7')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_bias_bars(bias_vals, band_names, save_path):
    x = np.arange(len(band_names))
    colors = ['#2166ac' if v < 0 else '#b2182b' for v in bias_vals]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, bias_vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Канал', fontsize=12)
    ax.set_ylabel('Смещение (отражение)', fontsize=12)
    ax.set_title('Среднее поканальное смещение', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, fontsize=10)
    ax.grid(True, axis='y', which='major', linestyle='-', linewidth=0.4, color='grey', alpha=0.4)
    ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    ax.grid(True, axis='y', which='minor', linestyle='-', linewidth=0.2, color='grey', alpha=0.2)
    ax.set_facecolor('#f7f7f7')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    from matplotlib.patches import Patch
    legend_elements = [
                    Patch(facecolor='#2166ac', label='Отрицательное'),
                    Patch(facecolor='#b2182b', label='Положительное')]
    ax.legend(handles=legend_elements, edgecolor='grey', fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_per_band_bars(orig_vals, sr_vals, band_names, ylabel, title, save_path, colors=('#2166ac', '#b2182b')):
    x = np.arange(len(band_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, orig_vals, width, label='Оригинал S2', color=colors[0], edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, sr_vals, width, label='S2DR3', color=colors[1], edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Канал', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, fontsize=10)
    ax.legend(fontsize=10, frameon=True, edgecolor='grey', facecolor='white')
    ax.grid(True, axis='y', which='major', linestyle='-', linewidth=0.4, color='grey', alpha=0.4)
    ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    ax.grid(True, axis='y', which='minor', linestyle='-', linewidth=0.2, color='grey', alpha=0.2)
    ax.set_facecolor('#f7f7f7')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_spectral_profiles(orig, sr, classes, band_names, out_dir, window=3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def extract_spectrum(stack, r, c, w):
        r0 = max(0, r - w//2)
        r1 = min(stack.shape[1], r + w//2 + 1)
        c0 = max(0, c - w//2)
        c1 = min(stack.shape[2], c + w//2 + 1)
        return np.nanmean(stack[:, r0:r1, c0:c1], axis=(1,2))

    n_bands = orig.shape[0]
    band_indices = np.arange(n_bands)

    all_vals = []
    for points in classes.values():
        for r, c in points:
            all_vals.append(extract_spectrum(orig, r, c, window))
            all_vals.append(extract_spectrum(sr, r, c, window))
    all_vals = np.concatenate(all_vals)
    y_min = np.min(all_vals) * 0.9
    y_max = np.max(all_vals) * 1.1

    for class_name, points in classes.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (r, c) in enumerate(points):
            spec_orig = extract_spectrum(orig, r, c, window)
            spec_sr = extract_spectrum(sr, r, c, window)
            lbl_orig = 'Оригинал' if i == 0 else None
            lbl_sr = 'SR' if i == 0 else None
            ax.plot(band_indices, spec_orig, 'o-', color='#2166ac',
                    label=lbl_orig)
            ax.plot(band_indices, spec_sr, 's--', color='#b2182b',
                    label=lbl_sr)
        ax.set_xticks(band_indices)
        ax.set_xticklabels(band_names, rotation=45)
        ax.set_xlabel('Канал')
        ax.set_ylabel('Отражение')
        ax.set_title(f'Спектральные профили — {class_name}')
        ax.set_ylim(y_min, y_max)
        ax.set_facecolor('#f7f7f7')
        ax.grid(True, axis='y', which='major', linestyle='-', linewidth=0.4, color='grey', alpha=0.4)
        ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
        ax.grid(True, axis='y', which='minor', linestyle='-', linewidth=0.2, color='grey', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', edgecolor='grey')
        fig.tight_layout()
        fig.savefig(out_dir / f'spectral_profile_{class_name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)