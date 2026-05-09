import logging
import yaml
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from typing import Dict, List, Tuple

def setup_logging(log_file=None, level=logging.INFO):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a'))
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    return logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_raster(filepath):
    with rasterio.open(filepath) as src:
        array = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    return array, transform, crs, bounds

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

def find_s2_band_files(directory: str, bands: List[str]) -> Dict[str, Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Директория {directory} не существует")
    band_files = {}
    for band in bands:
        found = list(directory.glob(f"*{band}*.jp2"))
        if not found:
            raise FileNotFoundError(f"Не найден файл для канала {band} в {directory}")
        band_files[band] = found[0]
    return band_files

def find_s2dr4_ms_file(directory: str) -> Path:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Директория {directory} не существует")
    ms_files = list(directory.glob("*_MS.tif"))
    if not ms_files:
        raise FileNotFoundError(f"Не найден файл *_MS.tif в {directory}")
    return ms_files[0]

def load_s2_bands_as_stack(band_files: Dict[str, Path], bands_order: List[str]) -> Tuple[np.ndarray, dict]:
    arrays = []
    transform = None
    crs = None
    bounds = None
    target_shape = None
    for band in bands_order:
        with rasterio.open(band_files[band]) as src:
            if target_shape is None:
                target_shape = src.shape
                transform = src.transform
                crs = src.crs
                bounds = src.bounds
            if src.shape != target_shape:
                arr = src.read(1, out_shape=target_shape, resampling=Resampling.bilinear).astype(np.float32)
            else:
                arr = src.read(1).astype(np.float32)
            arrays.append(arr)
    stack = np.stack(arrays, axis=0)
    geoinfo = {'transform': transform, 'crs': crs, 'bounds': bounds}
    return stack, geoinfo

def load_s2dr4_ms(filepath: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(filepath) as src:
        stack = src.read()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    geoinfo = {'transform': transform, 'crs': crs, 'bounds': bounds}
    return stack, geoinfo

def save_colormap_image(array, output_path, transform, crs, cmap='coolwarm', title=None, draw_grid=False, bounds=None, vmin=None, vmax=None):
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

def plot_radial_spectrum(frequencies, power_original, power_sr, save_path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.loglog(frequencies, power_original, label='Original S2 (10m)', linewidth=2)
    plt.loglog(frequencies, power_sr, label='Downscaled SR (10m)', linewidth=2)
    plt.xlabel('Spatial frequency (cycles/pixel)')
    plt.ylabel('Power spectral density')
    plt.title('Radial power spectrum')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()