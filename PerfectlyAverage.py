# -*- coding: utf-8 -*-
"""
PerfectlyAverage
Created on Thu Apr 24 14:50:58 2025
@author: Gail McConnell, SIPBS, University of Strathclyde, Glasgow, UK
"""

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from scipy.stats import entropy
from scipy.ndimage import uniform_filter1d
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import math
import time  # For timing execution

# Colour-blind-friendly colours
color_blue = "#0072B2"
color_orange = "#D55E00"
color_green = "#009E73"

# GUI for file selection and inputs
root = tk.Tk()
root.withdraw()

# Ask the user to select the TIFF file
input_path = filedialog.askopenfilename(
    title="Select a multi-frame TIFF file for averaging and analysis",
    filetypes=[("TIFF files", "*.tif;*.tiff")]
)

if not input_path:
    print("No file selected. Exiting.")
    exit()

# Load the image stack
stack = tiff.imread(input_path)
num_frames = stack.shape[0]
print(f"Loaded image stack with {num_frames} frames.")

# Determine max averaging level (powers of 2 ≤ number of frames)
max_exponent = int(math.floor(math.log2(num_frames)))
averaging_levels = [2**i for i in range(max_exponent + 1)]  # e.g., [1, 2, ..., 512, 1024]

# User input: Maximum allowed drop in fluorescence intensity (%)
while True:
    try:
        max_fluorescence_drop = simpledialog.askfloat(
            "Maximum Acceptable Fluorescence Decrease via Photobleaching (%)",
            "Enter 100 to include all frames (ignore photobleaching).",
            initialvalue=100,
            minvalue=0,
            maxvalue=100
        )
        if max_fluorescence_drop is None:
            raise ValueError("User canceled input.")
        min_intensity_ratio = 1 - (max_fluorescence_drop / 100)
        break 
    except (ValueError, TypeError):
        messagebox.showerror("Input Error", "Please enter a valid number between 0 and 100.")

# User input: PSD threshold
while True:
    try:
        psd_threshold = simpledialog.askfloat(
            "PSD Threshold",
            "Enter the cut-off value for PSD analysis.\n"
            "Recommended value: 0.1",
            initialvalue=0.1,
            minvalue=0.0,
            maxvalue=1.0
        )
        if psd_threshold is None:
            raise ValueError("User canceled input.")
        break
    except (ValueError, TypeError):
        messagebox.showerror("Input Error", "Please enter a valid number between 0 and 1.")

# ===== START TIMING EXECUTION =====
start_time = time.time()

# Compute averaged images
averaged_images = [np.mean(stack[:N], axis=0).astype(np.uint16) for N in averaging_levels]
print("Averaged images computed.")

# Helper functions
def calculate_snr(image):
    signal = np.mean(image)
    noise = np.std(image)
    return signal / noise if noise != 0 else 0

def compute_power_spectrum(image):
    return np.abs(fftshift(fft2(image))) ** 2

def compute_spectral_entropy(psd):
    flat = psd.flatten()
    prob = flat / np.sum(flat)
    return entropy(prob)

def normalise_values(values):
    min_val, max_val = np.min(values), np.max(values)
    return (values - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(values)

def find_stable_peak(values, window_size=5, threshold=0.1):
    smoothed = uniform_filter1d(values, size=window_size)
    diffs = np.abs(np.diff(smoothed))
    for i in range(1, len(diffs)):
        if diffs[i] < threshold:
            return i + 1
    return np.argmax(values) + 1

def plot_combined_results(intensities, snrs, psds, best_snr, best_psd):
    x = np.arange(len(intensities))
    plt.figure(figsize=(12, 7))
    
    # Plot normalized values
    plt.plot(x, normalise_values(intensities), 'o-', linewidth=3, label="Mean Intensity", color=color_blue)
    plt.plot(x, normalise_values(snrs), 's--', linewidth=3, label="SNR", color=color_orange)
    plt.plot(x, normalise_values(psds), '^-.', linewidth=3, label="PSD", color=color_green)
    
    # Vertical lines for best choices
    plt.axvline(best_snr - 1, linestyle=':', linewidth=4, color=color_orange,
                label="Optimal averaging based on SNR within photobleaching limit")
    plt.axvline(best_psd - 1, linestyle=':', linewidth=4, color=color_green,
                label="Optimal averaging based on PSD")
    
    # Fix superscripts for 2^i labels
    xtick_labels = [f"$2^{{{i}}}$" for i in range(len(x))]
    plt.xticks(x, xtick_labels, fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlabel("Number of Averages", fontsize=16)
    plt.ylabel("Normalised Value", fontsize=16)
    plt.title("PerfectlyAverage Analysis", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Perform analysis
mean_intensities = [np.mean(im) for im in averaged_images]
snr_values = [calculate_snr(im) for im in averaged_images]
spectral_entropies = [compute_spectral_entropy(compute_power_spectrum(im)) for im in averaged_images]
norm_psd = normalise_values(spectral_entropies)

valid_frames = np.where(normalise_values(mean_intensities) >= min_intensity_ratio)[0]
best_snr_frame = valid_frames[np.argmax([snr_values[i] for i in valid_frames])] + 1 if valid_frames.size > 0 else 1
best_psd_frame = find_stable_peak(norm_psd, window_size=5, threshold=psd_threshold)

# Plot results
plot_combined_results(mean_intensities, snr_values, norm_psd, best_snr_frame, best_psd_frame)

# Save the result
output_path = input_path.replace(".tif", " averaged.tif")
tiff.imwrite(output_path, averaged_images)
print(f"Saved averaged images to {output_path}")

# ===== END TIMING EXECUTION =====
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time (analysis and save only): {elapsed_time:.2f} seconds")

# Show message box with results
summary = (
    f"Optimal number of averages based on SNR within photobleaching limit: {2**(best_snr_frame - 1)}\n"
    f"Optimal number of averages based on PSD: {2**(best_psd_frame - 1)}\n"
)
messagebox.showinfo("PerfectlyAverage Results", summary)
