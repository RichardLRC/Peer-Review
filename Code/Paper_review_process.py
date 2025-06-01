import os
import json
from tqdm import tqdm
import re

class ReviewProcessor:
    def __init__(self, root_dir):
        """Initialize with the root data directory"""
        self.root_dir = root_dir

        # Define years for each venue
        self.venue_years = {
            "ICLR": ["2024", "2025"],
            "NeurIPS": ["2023", "2024"]
        }

        self.labels = ["good", "borderline", "bad"]

    def parse_score(self, value):
        """Extract numeric score from strings like '4: excellent' or '2 fair'"""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.match(r"^\s*(\d+)", value)
            if match:
                return int(match.group(1))
        return None

    def process_all(self):
        """Process reviews for all conferences and years"""
        for conf, years in self.venue_years.items():
            for year in years:
                self.process_one(conf, year)
        print("\n🎉 All real reviews have been processed successfully.")

    def process_one(self, conf, year):
        """Process reviews for a specific conference and year"""
        paper_dir = os.path.join(self.root_dir, conf, year, "papers")
        real_review_dir = os.path.join(self.root_dir, conf, year, "real_review")

        for label in self.labels:
            input_path = os.path.join(paper_dir, f"{label}_papers_reviews.json")
            output_folder = os.path.join(real_review_dir, f"{label}_papers")
            os.makedirs(output_folder, exist_ok=True)

            if not os.path.exists(input_path):
                print(f"⚠️ File not found, skipping: {input_path}")
                continue

            print(f"\n📄 Processing: {input_path}")
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    paper_data = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ Failed to parse JSON: {input_path}")
                continue

            formatted_reviews = []

            for paper in tqdm(paper_data, desc=f"📝 {conf}{year} - {label}"):
                paper_id = paper.get("paper_id", "unknown")
                title = paper.get("title", "Untitled")
                reviews = []

                for reply in paper.get("all_replies", []):
                    content = reply.get("content", {})
                    review_id = reply.get("id", "unknown")

                    # Only process reviews that contain all required fields
                    if "summary" in content and "soundness" in content:
                        try:
                            review_data = {
                                "review_id": review_id,
                                "summary": content["summary"]["value"],
                                "soundness": self.parse_score(content["soundness"]["value"]),
                                "presentation": self.parse_score(content["presentation"]["value"]),
                                "contribution": self.parse_score(content["contribution"]["value"]),
                                "strengths": content["strengths"]["value"],
                                "weaknesses": content["weaknesses"]["value"],
                                "questions": content["questions"]["value"],
                                "overall_rating": self.parse_score(content["rating"]["value"]),
                                "confidence": self.parse_score(content["confidence"]["value"])
                            }
                            reviews.append(review_data)
                        except Exception as e:
                            print(f"⚠️ Failed to parse review {review_id}: {e}")

                if reviews:
                    formatted_reviews.append({
                        "paper_id": paper_id,
                        "title": title,
                        "reviews": reviews
                    })

            output_path = os.path.join(output_folder, f"{label}_reviews.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(formatted_reviews, f, indent=2, ensure_ascii=False)

            print(f"✅ Saved to: {output_path}")


# ✅ Entry point
if __name__ == "__main__":
    processor = ReviewProcessor("/root/Peer-Review-LLMs/Data")
    processor.process_all()
