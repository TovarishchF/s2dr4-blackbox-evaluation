import logging
import yaml
import numpy as np
import rasterio
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
        found = list(directory.glob(f"*{band}*.tif*"))
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
    if len(ms_files) > 1:
        logging.warning(f"Найдено несколько MS файлов: {ms_files}, беру первый")
    return ms_files[0]


def load_s2_bands_as_stack(band_files: Dict[str, Path], bands_order: List[str]) -> Tuple[np.ndarray, dict]:
    arrays = []
    transform = None
    crs = None
    for band in bands_order:
        with rasterio.open(band_files[band]) as src:
            arr = src.read(1).astype(np.float32)
            arrays.append(arr)
            if transform is None:
                transform = src.transform
                crs = src.crs
            if arr.shape != arrays[0].shape:
                raise ValueError(f"Размеры канала {band} {arr.shape} не совпадают с {arrays[0].shape}")
    stack = np.stack(arrays, axis=0)
    geoinfo = {'transform': transform, 'crs': crs}
    return stack, geoinfo


def load_s2dr4_ms(filepath: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(filepath) as src:
        stack = src.read()
        transform = src.transform
        crs = src.crs
    geoinfo = {'transform': transform, 'crs': crs}
    return stack, geoinfo