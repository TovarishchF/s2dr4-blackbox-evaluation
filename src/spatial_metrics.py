import numpy as np
from skimage.feature import graycomatrix, graycoprops, canny
from skimage import filters
from skimage.util import img_as_float
from skimage.filters import threshold_otsu


def glcm_contrast(image, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=16):
    if image.dtype.kind == 'f':
        img = (image * (levels - 1)).astype(np.int32)
        img = np.clip(img, 0, levels - 1)
    else:
        max_val = image.max()
        if max_val == 0:
            img = np.zeros_like(image, dtype=np.uint8)
        else:
            if max_val >= levels:
                img = (image * (levels - 1) / max_val).astype(np.int32)
            else:
                img = image.astype(np.int32)
            img = np.clip(img, 0, levels - 1)
    img = img.astype(np.uint8)

    glcm = graycomatrix(img, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    contrast_all = graycoprops(glcm, 'contrast')
    return np.mean(contrast_all)


def glcm_contrast_multiband(stack, distances=[1], angles=None, levels=16):
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    contrasts = []
    for b in range(stack.shape[0]):
        contrasts.append(glcm_contrast(stack[b], distances, angles, levels))
    return np.array(contrasts)


def edge_density(image, method='canny', low_threshold=None, high_threshold=None, sigma=1.0):
    img = img_as_float(image)
    if method == 'canny':
        if low_threshold is None or high_threshold is None:
            low_threshold, high_threshold = 0.1, 0.3
        edges = canny(img, sigma=sigma,
                      low_threshold=low_threshold,
                      high_threshold=high_threshold)
    elif method == 'sobel':
        grad = filters.sobel(img)
        thresh = threshold_otsu(grad)
        edges = grad > thresh
    else:
        raise ValueError("method must be 'canny' or 'sobel'")
    return np.mean(edges)


def edge_density_multiband(stack, method='canny', **kwargs):
    densities = []
    for b in range(stack.shape[0]):
        densities.append(edge_density(stack[b], method=method, **kwargs))
    return np.array(densities)