import os
import pandas as pd
import requests
from tqdm import tqdm

def download_pdf(paper_id, save_dir):
    """Download PDF; skip if already exists"""
    url = f"https://openreview.net/pdf?id={paper_id}"
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f"{paper_id}.pdf")

    if os.path.exists(pdf_path):
        return "skipped"  # Already exists

    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            return "success"
        else:
            return f"status_{response.status_code}"
    except Exception as e:
        return f"error_{str(e)}"

def download_from_labeled_file(csv_path):
    """Read labeled CSV file and download papers by category"""
    df = pd.read_csv(csv_path)
    root_dir = os.path.dirname(csv_path)
    failure_log_path = os.path.join(root_dir, "download_failures.txt")

    # Clear previous failure log
    open(failure_log_path, "w").close()

    label_groups = df.groupby("label")
    for label, group in label_groups:
        paper_ids = group["id"].tolist()
        save_dir = os.path.join(root_dir, label)

        print(f"\n Downloading: {os.path.basename(csv_path)} - {label} ({len(paper_ids)} papers)")
        for paper_id in tqdm(paper_ids, desc=f"{label}"):
            result = download_pdf(paper_id, save_dir)
            if result == "success":
                tqdm.write(f"Success: {paper_id}")
            elif result == "skipped":
                tqdm.write(f"Skipped: {paper_id}")
            else:
                tqdm.write(f"Failed: {paper_id} ({result})")
                with open(failure_log_path, "a") as log_f:
                    log_f.write(f"{paper_id},{label},{result}\n")

def batch_download(file_list):
    for csv_file in file_list:
        if os.path.exists(csv_file):
            download_from_labeled_file(csv_file)
        else:
            print(f"File not found: {csv_file}")

# === Labeled CSV file list ===
file_list = [
    "../Data/ICLR/2025/papers/ICLR2025_papers_avg_rating_consistent_labeled.csv",
    "../Data/ICLR/2024/papers/ICLR2024_papers_avg_rating_consistent_labeled.csv",
    "../Data/NeurIPS/2023/papers/NeurIPS2023_papers_avg_rating_consistent_labeled.csv",
    "../Data/NeurIPS/2024/papers/NeurIPS2024_papers_avg_rating_consistent_labeled.csv"
]

# === Execute download ===
batch_download(file_list)
