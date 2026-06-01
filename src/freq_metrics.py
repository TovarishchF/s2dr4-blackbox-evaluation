import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

from src.locale import get

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

def plot_radial_spectrum(freqs, power_orig, power_sr, save_path=None,
                         title=None, xlabel=None, ylabel=None,
                         label_orig=None, label_sr=None,
                         xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    if title is None:
        title = get('plot_title_radial')
    if xlabel is None:
        xlabel = get('plot_xlabel_freq')
    if ylabel is None:
        ylabel = get('plot_ylabel_power')
    if label_orig is None:
        label_orig = get('label_original')
    if label_sr is None:
        label_sr = get('label_sr')

    ax.loglog(freqs, power_orig, label=label_orig, linewidth=1.6, color='#2166ac')
    ax.loglog(freqs, power_sr, label=label_sr, linewidth=1.6, color='#b2182b')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, frameon=True, edgecolor='none', facecolor='white', loc='best')
    ax.grid(True, which='major', linestyle='-', linewidth=0.4, color='grey', alpha=0.4)
    ax.set_facecolor('#f7f7f7')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_spectrum_diff(freqs, power_orig, power_sr, save_path=None,
                       title=None, xlabel=None, ylabel=None,
                       xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    if title is None:
        title = get('plot_title_diff')
    if xlabel is None:
        xlabel = get('plot_xlabel_freq')
    if ylabel is None:
        ylabel = get('plot_ylabel_diff')

    diff = np.log10(power_sr + 1e-12) - np.log10(power_orig + 1e-12)
    ax.semilogx(freqs, diff, linewidth=1.6, color='#9970ab')
    ax.axhline(y=0, linestyle='--', color='grey', alpha=0.7, linewidth=1.0)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='major', linestyle='-', linewidth=0.4, color='grey', alpha=0.4)
    ax.set_facecolor('#f7f7f7')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()