import pandas as pd
import numpy as np
import os

# Map each file to its std_rating threshold (manually determined inflection point)
threshold_dict = {
    "ICLR2025_papers_avg_rating.csv": 0.66,
    "ICLR2024_papers_avg_rating.csv": 0.66,
    "NeurIPS2023_papers_avg_rating.csv": 0.61,
    "NeurIPS2024_papers_avg_rating.csv": 0.62
}

def extract_consistent_papers(file_path, label_percent=0.025, borderline_window=0.0125):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    std_threshold = threshold_dict.get(os.path.basename(file_path), 1.0)
    df = pd.read_csv(file_path)

    # Filter consistent papers by std_rating threshold
    df_consistent = df[df['std_rating'] <= std_threshold].copy()

    # Compute quantile-based thresholds using the full dataset
    good_threshold = df["average_rating"].quantile(1 - label_percent)
    bad_threshold = df["average_rating"].quantile(label_percent)
    borderline_min = df["average_rating"].quantile(0.5 - borderline_window)
    borderline_max = df["average_rating"].quantile(0.5 + borderline_window)

    # Assign labels
    df_consistent['label'] = 'none'
    df_consistent.loc[df_consistent['average_rating'] >= good_threshold, 'label'] = 'good'
    df_consistent.loc[df_consistent['average_rating'] <= bad_threshold, 'label'] = 'bad'
    df_consistent.loc[
        (df_consistent['average_rating'] >= borderline_min) &
        (df_consistent['average_rating'] <= borderline_max),
        'label'
    ] = 'borderline'

    # Select only labeled samples
    df_labeled = df_consistent[df_consistent['label'] != 'none']

    # Save result
    output_path = f"{os.path.dirname(file_path)}/{dataset_name}_consistent_labeled.csv"
    df_labeled.to_csv(output_path, index=False)

    # Return path and label distribution
    label_counts = df_labeled['label'].value_counts().to_dict()
    ordered_labels = ['borderline', 'bad', 'good']
    label_str = ", ".join([f"{label}: {label_counts.get(label, 0)}" for label in ordered_labels])
    print(f"✅ {dataset_name:<35} (std ≤ {std_threshold:<4}): {label_str}")
    return output_path, label_counts

# === Batch processing ===
file_list = [
    "../Data/ICLR/2025/papers/ICLR2025_papers_avg_rating.csv",
    "../Data/ICLR/2024/papers/ICLR2024_papers_avg_rating.csv",
    "../Data/NeurIPS/2023/papers/NeurIPS2023_papers_avg_rating.csv",
    "../Data/NeurIPS/2024/papers/NeurIPS2024_papers_avg_rating.csv"
]

results = []
for file_path in file_list:
    if os.path.exists(file_path):
        results.append(extract_consistent_papers(file_path))
    else:
        print(f"❌ File not found: {file_path}")
