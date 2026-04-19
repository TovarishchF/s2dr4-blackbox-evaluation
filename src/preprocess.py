import numpy as np
from skimage.transform import resize
from src.utils import (
    load_s2_bands_as_stack, load_s2dr4_ms, find_s2_band_files, find_s2dr4_ms_file
)

def load_data(config):
    s2_dir = config['data']['s2_dir']
    s2dr4_dir = config['data']['s2dr4_dir']
    bands_order = config['bands']

    band_files = find_s2_band_files(s2_dir, bands_order)
    orig_stack, orig_geoinfo = load_s2_bands_as_stack(band_files, bands_order)

    s2dr4_path = find_s2dr4_ms_file(s2dr4_dir)
    sr_stack, sr_geoinfo = load_s2dr4_ms(s2dr4_path)

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


if __name__ == "__main__":
    import sys
    import os
    import logging
    from pathlib import Path

    root_dir = Path(__file__).parent.parent
    os.chdir(root_dir)
    sys.path.insert(0, str(root_dir))

    from src.utils import load_config, setup_logging

    logger = setup_logging(log_file="logs/preprocess_test.log", level=logging.INFO)
    logger.info("Запуск preprocess.py")

    config = load_config("config.yaml")

    try:
        orig_stack, sr_stack, orig_geoinfo, sr_geoinfo = load_data(config)
        logger.info(f"Оригинальный S2 стек: форма {orig_stack.shape}, dtype {orig_stack.dtype}")
        logger.info(f"S2DR4 стек: форма {sr_stack.shape}, dtype {sr_stack.dtype}")

        target_shape = orig_stack.shape[1:]
        downsampled = downsample_sr_to_original(sr_stack, target_shape)
        logger.info(f"Downsampled SR форма: {downsampled.shape}")

        assert downsampled.shape == orig_stack.shape, "Размеры не совпадают!"

        logger.info("Тест preprocess.py пройден успешно.")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise