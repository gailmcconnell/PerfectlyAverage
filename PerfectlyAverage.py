# -*- coding: utf-8 -*-
"""
Created on Friday 7 March 07:50:58 2025

@author: Gail McConnell, University of Strathclyde
"""

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fftpack import fft2, fftshift
from scipy.stats import entropy
from scipy.ndimage import uniform_filter1d
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog

# Define colour vision deficiency-friendly colours
color_blue = "#0072B2"   # Blue
color_orange = "#D55E00" # Orange
color_green = "#009E73"  # Green

# File selection dialog
root = tk.Tk()
root.withdraw()
image_stack_path = filedialog.askopenfilename(
    title="Select TIFF file",
    filetypes=[("TIFF files", "*.tif;*.tiff")]
)

# Select user-defined acceptable photobleaching amount (%)
while True:
    try:
        photobleaching_input = simpledialog.askfloat(
            "Photobleaching Limit", 
            "Enter acceptable intensity drop-off in % (e.g., 10 for 10%)",
            minvalue=0, maxvalue=100
        )
        if photobleaching_input is None:
            raise ValueError("User canceled input.")
        photobleaching_limit = 1 - (photobleaching_input / 100)
        break 
    except (ValueError, TypeError):
        messagebox.showerror("Input Error", "Please enter a valid numerical value between 0 and 100.")

# Functions for analysis
def calculate_snr(image):
    signal = np.mean(image)
    noise = np.std(image)
    return signal / noise if noise != 0 else 0

def exponential_decay(x, a, b, c):
    max_exp = 700
    return a * np.exp(np.clip(-b * x, -max_exp, max_exp)) + c

def correct_photobleaching(image_stack):
    mean_intensities = [np.mean(image) for image in image_stack]
    frame_indices = np.arange(len(mean_intensities)) + 1  
    popt, _ = curve_fit(exponential_decay, frame_indices, mean_intensities, maxfev=5000)
    
    corrected_stack = []
    for idx, image in enumerate(image_stack):
        correction_factor = exponential_decay(frame_indices[idx], *popt) / mean_intensities[idx]
        corrected_stack.append(image * correction_factor)
    
    return np.array(corrected_stack), mean_intensities

def compute_power_spectrum(image):
    fft_image = fft2(image)
    fft_shifted = fftshift(fft_image)  
    return np.abs(fft_shifted) ** 2  

def compute_spectral_entropy(psd):
    psd_flat = psd.flatten()
    psd_prob = psd_flat / np.sum(psd_flat) 
    return entropy(psd_prob)

def normalise_values(values):
    min_val, max_val = np.min(values), np.max(values)
    return (values - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(values)

def find_stable_peak(values, window_size=5, threshold=0.1):
    smoothed_values = uniform_filter1d(values, size=window_size)
    diffs = np.abs(np.diff(smoothed_values))  
    for i in range(1, len(diffs)):
        if diffs[i] < threshold:  
            return i + 1  
    return np.argmax(values) + 1  

def analyse_frequency_distributions(image_stack):
    spectral_entropies = [compute_spectral_entropy(compute_power_spectrum(image)) for image in image_stack]
    norm_spectral_entropies = normalise_values(spectral_entropies)
    return find_stable_peak(norm_spectral_entropies, window_size=5, threshold=0.1), norm_spectral_entropies

def plot_combined_results(mean_intensities, snr_values, norm_spectral_entropies, best_snr_frame, best_frequency_frame):
    num_images = len(mean_intensities)
    x_positions = np.arange(num_images)
    norm_mean_intensities = normalise_values(mean_intensities)
    norm_snr_values = normalise_values(snr_values)
    
    plt.figure(figsize=(12, 7))
    plt.plot(x_positions, norm_mean_intensities, marker='o', markersize=8, linestyle='-', linewidth=3, label="Mean Intensity", color=color_blue)
    plt.plot(x_positions, norm_snr_values, marker='s', markersize=8, linestyle='--', linewidth=3, label="SNR", color=color_orange)
    plt.plot(x_positions, norm_spectral_entropies, marker='^', markersize=8, linestyle='-.', linewidth=3, label="PSD", color=color_green)
    
    plt.axvline(best_snr_frame - 1, color=color_orange, linestyle=':', linewidth=4, alpha=0.9, label="Optimal averaging based on SNR")
    plt.axvline(best_frequency_frame - 1, color=color_green, linestyle=':', linewidth=4, alpha=0.9, label="Optimal averaging based on PSD")
    
    plt.xlabel("Number of Averages", fontsize=16)
    plt.ylabel("Normalised Value", fontsize=16)
    plt.title("PerfectlyAverage: Mean Intensity, SNR, and PSD", fontsize=16)
    
    power_labels = [rf"$2^{{{i}}}$" for i in range(num_images)] 
    plt.xticks(x_positions, power_labels, fontsize=16)  
    plt.yticks(fontsize=16)  
    
    plt.legend(loc="best", fontsize=16)
    plt.grid(linewidth=1.2) 
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not image_stack_path:
        print("No file selected. Exiting.")
        exit()

    image_stack = tiff.imread(image_stack_path)
    corrected_stack, mean_intensities = correct_photobleaching(image_stack)
    snr_values = [calculate_snr(image) for image in corrected_stack]
    best_frequency_frame, norm_spectral_entropies = analyse_frequency_distributions(image_stack)
    
    valid_frames = np.where(normalise_values(mean_intensities) >= photobleaching_limit)[0]
    best_snr_frame = valid_frames[np.argmax(normalise_values([snr_values[i] for i in valid_frames]))] + 1 if len(valid_frames) > 0 else 1
    
    plot_combined_results(mean_intensities, snr_values, norm_spectral_entropies, best_snr_frame, best_frequency_frame)
    
    confirmation_message = (
        f"Optimal number of averages based on SNR & photobleaching: {2**(best_snr_frame - 1)}\n"       
        f"Optimal number of averages based on PSD: {2**(best_frequency_frame - 1)}\n"
        f"Photobleaching Limit Applied: {100 - (photobleaching_limit * 100):.2g}%"
    )
    messagebox.showinfo("PerfectlyAverage: Optimal Image Selection Results", confirmation_message)
