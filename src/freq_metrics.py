import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

def radial_spectrum(image, pixel_size=10.0, return_freqs=True):
    h, w = image.shape
    f = fft2(image)
    fshift = fftshift(f)
    power = np.abs(fshift) ** 2

    cx, cy = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r.astype(int)

    max_radius = min(cx, cy)
    radial_means = []
    for rad in range(1, max_radius):
        mask = (r == rad)
        radial_means.append(np.mean(power[mask]))
    radial_means = np.array(radial_means)

    freqs_pix = np.linspace(0.5 / max_radius, 0.5, len(radial_means))
    if return_freqs:
        freqs_m = freqs_pix / pixel_size
        return freqs_m, radial_means
    else:
        return freqs_pix, radial_means

def plot_radial_spectrum(freqs, power_orig, power_sr, save_path=None, title=None):
    plt.figure(figsize=(8, 6))
    plt.loglog(freqs, power_orig, label='Оригинал S2 (10 м)', linewidth=1.6)
    plt.loglog(freqs, power_sr, label='S2DR4 (10 м)', linewidth=1.6)
    plt.xlabel('Пространственная частота (циклов/м)', fontsize=12)
    plt.ylabel('Спектральная плотность мощности', fontsize=12)
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title('Радиальный спектр мощности', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_radial_spectrum_ratio(freqs, power_orig, power_sr, save_path=None, title=None):
    plt.figure(figsize=(8, 6))
    ratio = power_sr / (power_orig + 1e-12)
    plt.semilogx(freqs, ratio, linewidth=1.6, color='green')
    plt.axhline(y=1, linestyle='--', color='gray', alpha=0.7)
    plt.xlabel('Пространственная частота (циклов/м)', fontsize=12)
    plt.ylabel('Отношение мощности (SR / Оригинал)', fontsize=12)
    if title:
        plt.title(f'{title} — Отношение спектров', fontsize=14)
    else:
        plt.title('Отношение радиальных спектров', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_spectrum_diff(freqs, power_orig, power_sr, save_path=None):
    diff = np.log10(power_sr + 1e-12) - np.log10(power_orig + 1e-12)
    plt.figure(figsize=(8, 6))
    plt.semilogx(freqs, diff, linewidth=1.6, color='blue')
    plt.axhline(y=0, linestyle='--', color='gray', alpha=0.7)
    plt.xlabel('Пространственная частота (циклов/м)', fontsize=12)
    plt.ylabel('Разность: log10(P_SR) - log10(P_orig)', fontsize=12)
    plt.title('Разность спектров (положительное = SR сильнее)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()