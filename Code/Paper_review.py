import openreview
import pandas as pd
import json
import os
from tqdm import tqdm

OPENREVIEW_USERNAME = "Your_Username"
OPENREVIEW_PASSWORD = "Your_Password"

# Initialize OpenReview client
def init_client():
    print("🔐 Initializing OpenReview client...")
    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=OPENREVIEW_USERNAME,
        password=OPENREVIEW_PASSWORD,
    )
    return client

# Fetch and save reviews by label
def fetch_reviews_and_save(client, df, label, save_dir):
    all_reviews = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"💬 Fetching {label} reviews"):
        paper_id = row["id"]
        title = row["title"]
        try:
            paper_notes = client.get_all_notes(forum=paper_id)
            all_replies = [note.to_json() for note in paper_notes]
        except Exception as e:
            print(f"❌ Failed to fetch: {paper_id} - {e}")
            continue

        all_reviews.append({
            "paper_id": paper_id,
            "title": title,
            "all_replies": all_replies
        })

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{label}_papers_reviews.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_reviews, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved to: {out_path}")

# Main process for a CSV file
def process_csv(client, csv_path):
    print(f"\n📄 Processing file: {csv_path}")
    df = pd.read_csv(csv_path)
    save_dir = os.path.dirname(csv_path)

    for label in ["good", "borderline", "bad"]:
        df_label = df[df["label"] == label]
        if not df_label.empty:
            fetch_reviews_and_save(client, df_label, label, save_dir)

# === Entry point ===
if __name__ == "__main__":
    csv_paths = [
        "../Data/ICLR/2025/papers/ICLR2025_papers_avg_rating_consistent_labeled.csv",
        "../Data/ICLR/2024/papers/ICLR2024_papers_avg_rating_consistent_labeled.csv",
        "../Data/NeurIPS/2023/papers/NeurIPS2023_papers_avg_rating_consistent_labeled.csv",
        "../Data/NeurIPS/2024/papers/NeurIPS2024_papers_avg_rating_consistent_labeled.csv"
    ]

    client = init_client()
    for csv_path in csv_paths:
        process_csv(client, csv_path)
