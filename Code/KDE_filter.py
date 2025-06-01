import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import os

# Output directory for filtered files
SAVE_DIR = "./filtered_papers"
os.makedirs(SAVE_DIR, exist_ok=True)

# KDE + valley detection and save filtered results
def detect_and_save_valley_filtered(df, title, file_name, n_valleys=2, bandwidth='scott'):
    std_values = df['std_rating'].dropna().values

    kde = gaussian_kde(std_values, bw_method=bandwidth)
    x_grid = np.linspace(0, std_values.max() + 0.5, 500)
    kde_values = kde(x_grid)

    # Detect valleys
    inverted_kde = -kde_values
    valley_indices, _ = find_peaks(inverted_kde, distance=10)
    valleys = x_grid[valley_indices[:n_valleys]] if len(valley_indices) >= n_valleys else x_grid[valley_indices]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.hist(std_values, bins=30, density=True, alpha=0.4, label='Histogram (std_rating)')
    plt.plot(x_grid, kde_values, label='KDE')
    for i, vx in enumerate(valleys):
        plt.axvline(vx, color='orange', linestyle='--', label=f"Valley {i+1}: {vx:.2f}")
    plt.title(f"KDE-based std_rating Threshold ({title})")
    plt.xlabel("std_rating")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save filtered data based on the second valley
    if len(valleys) >= 2:
        threshold = valleys[1]
        df_filtered = df[df["std_rating"] <= threshold].copy()
        save_path = os.path.join(SAVE_DIR, file_name)
        df_filtered.to_csv(save_path, index=False)
        print(f"{title}: {len(df_filtered)} papers saved to {save_path} (threshold = {threshold:.3f})")
    else:
        print(f"{title}: Not enough valleys detected.")

    return valleys

# Load data
df_iclr_2025 = pd.read_csv("../Data/ICLR/2025/papers/ICLR2025_papers_avg_rating.csv")
df_iclr_2024 = pd.read_csv("../Data/ICLR/2024/papers/ICLR2024_papers_avg_rating.csv")
df_neurips_2023 = pd.read_csv("../Data/NeurIPS/2023/papers/NeurIPS2023_papers_avg_rating.csv")
df_neurips_2024 = pd.read_csv("../Data/NeurIPS/2024/papers/NeurIPS2024_papers_avg_rating.csv")

# Run detection and save filtered results
valleys_iclr_2025 = detect_and_save_valley_filtered(df_iclr_2025, "ICLR2025", "ICLR2025_filtered.csv")
valleys_iclr_2024 = detect_and_save_valley_filtered(df_iclr_2024, "ICLR2024", "ICLR2024_filtered.csv")
valleys_neurips_2023 = detect_and_save_valley_filtered(df_neurips_2023, "NeurIPS2023", "NeurIPS2023_filtered.csv")
valleys_neurips_2024 = detect_and_save_valley_filtered(df_neurips_2024, "NeurIPS2024", "NeurIPS2024_filtered.csv")

# Print thresholds
print("std_thresholds by 2nd KDE valley:")
print(f"ICLR2025: {valleys_iclr_2025[1] if len(valleys_iclr_2025) >= 2 else 'N/A'}")
print(f"ICLR2024: {valleys_iclr_2024[1] if len(valleys_iclr_2024) >= 2 else 'N/A'}")
print(f"NeurIPS2023: {valleys_neurips_2023[1] if len(valleys_neurips_2023) >= 2 else 'N/A'}")
print(f"NeurIPS2024: {valleys_neurips_2024[1] if len(valleys_neurips_2024) >= 2 else 'N/A'}")
