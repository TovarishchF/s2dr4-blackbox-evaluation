import numpy as np
import rasterio
from skimage.transform import resize
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling

from src.utils import load_stack_ms, find_stack_ms_file


def load_data(territory_cfg):
    s2_dir = territory_cfg['s2_dir']
    s2sr_dir = territory_cfg['s2sr_dir']
    band_files = find_stack_ms_file(s2_dir)
    orig_stack, orig_geoinfo = load_stack_ms(band_files)
    s2sr_path = find_stack_ms_file(s2sr_dir)
    sr_stack, sr_geoinfo = load_stack_ms(s2sr_path)
    return orig_stack, sr_stack, orig_geoinfo, sr_geoinfo

def downsample_sr_to_original(sr_stack, target_shape, method='bilinear'):
    bands = sr_stack.shape[0]
    downsampled = np.zeros((bands, target_shape[0], target_shape[1]), dtype=sr_stack.dtype)
    for b in range(bands):
        downsampled[b] = resize(
            sr_stack[b], target_shape, order=1 if method == 'bilinear' else 3,
            preserve_range=True, anti_aliasing=True
        )
    return downsampled

def crop_to_common_extent(orig_stack, orig_geoinfo, sr_stack, sr_geoinfo):
    left = max(orig_geoinfo['bounds'].left, sr_geoinfo['bounds'].left)
    bottom = max(orig_geoinfo['bounds'].bottom, sr_geoinfo['bounds'].bottom)
    right = min(orig_geoinfo['bounds'].right, sr_geoinfo['bounds'].right)
    top = min(orig_geoinfo['bounds'].top, sr_geoinfo['bounds'].top)

    if left >= right or bottom >= top:
        raise ValueError("Изображения не перекрываются!")

    window_orig = from_bounds(left, bottom, right, top, transform=orig_geoinfo['transform'])
    row_start = int(round(window_orig.row_off))
    row_stop  = int(round(window_orig.row_off + window_orig.height))
    col_start = int(round(window_orig.col_off))
    col_stop  = int(round(window_orig.col_off + window_orig.width))
    orig_cropped = orig_stack[:, row_start:row_stop, col_start:col_stop]
    height = row_stop - row_start
    width  = col_stop - col_start

    new_transform = rasterio.Affine(
        orig_geoinfo['transform'].a, orig_geoinfo['transform'].b, left,
        orig_geoinfo['transform'].d, orig_geoinfo['transform'].e, top
    )

    dst_array = np.zeros((sr_stack.shape[0], height, width), dtype=np.float32)
    reproject(
        source=sr_stack,
        destination=dst_array,
        src_transform=sr_geoinfo['transform'],
        src_crs=sr_geoinfo['crs'],
        dst_transform=new_transform,
        dst_crs=orig_geoinfo['crs'],
        resampling=Resampling.bilinear
    )

    common_geoinfo = {
        'transform': new_transform,
        'crs': orig_geoinfo['crs'],
        'bounds': (left, bottom, right, top)
    }
    return orig_cropped, dst_array, common_geoinfo