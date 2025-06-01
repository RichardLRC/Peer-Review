import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Load BGE-M3 model
model = SentenceTransformer("BAAI/bge-m3")

# Configuration
ROOT_DIR = "../Data"
conferences_years = {
    "ICLR": ["2024", "2025"],
    "NeurIPS": ["2023", "2024"]
}
categories = ["good", "borderline", "bad"]
llm_models = ["gpt", "gemini", "claude", "llama", "qwen"]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def encode_sections(section_file):
    """Encode each mapped section of a paper"""
    sections = load_json(section_file)
    section_texts = {sec["mapped_section"]: sec["content"] for sec in sections}
    section_embeddings = {
        sec: model.encode(text, convert_to_tensor=True)
        for sec, text in section_texts.items()
    }
    return section_embeddings

def compute_similarity(part_text, section_embeddings):
    """Compute similarity between a review part and each section"""
    if not part_text.strip():
        return {}
    part_embedding = model.encode(part_text, convert_to_tensor=True)
    similarities = {
        section: util.pytorch_cos_sim(part_embedding, embedding).item()
        for section, embedding in section_embeddings.items()
    }
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [(sec, round(score, 4)) for sec, score in sorted_similarities]

def process_reviews(paper_id, section_embeddings, reviews, review_source):
    """Process all reviews for a given paper"""
    similarity_reviews = {"paper_id": paper_id, "reviews": []}

    for review in reviews:
        if review_source == "real":
            review_parts = {
                "summary": review.get("summary", ""),
                "strengths": review.get("strengths", ""),
                "weaknesses": review.get("weaknesses", ""),
                "questions": review.get("questions", "")
            }
        else:  # LLM reviews
            review_parts = {
                "summary": review.get("summary", ""),
                "strengths": review.get("strengths", ""),
                "weaknesses": review.get("weaknesses", ""),
                "questions": " ".join(review.get("questions", [])),  # questions is a list
                "reasons_for_overall_rating": review.get("Reasons for overall_rating", review.get("Reasons for overal_rating", ""))
            }

        similarity_data = {}
        for part_name, part_text in review_parts.items():
            if isinstance(part_text, list):
                part_text = " ".join(part_text)

            if isinstance(part_text, str) and part_text.strip():
                similarity_data[part_name] = compute_similarity(part_text, section_embeddings)

        similarity_reviews["reviews"].append(similarity_data)

    return similarity_reviews

def main():
    for conf, years in conferences_years.items():
        for year in years:
            print(f"\n Processing {conf} {year}...")
            for category in categories:
                print(f" Category: {category}")

                # Define paths
                section_map_dir = os.path.join(ROOT_DIR, conf, year, "md_section_paper", category)
                real_review_file = os.path.join(ROOT_DIR, conf, year, "real_review", f"{category}_papers", f"{category}_reviews.json")
                real_output_dir = os.path.join(ROOT_DIR, conf, year, "similarity_results", "real_review", category)

                llm_review_dirs = {
                    model_name: os.path.join(ROOT_DIR, conf, year, f"{model_name}_review", f"{category}_papers")
                    for model_name in llm_models
                }
                llm_output_dirs = {
                    model_name: os.path.join(ROOT_DIR, conf, year, "similarity_results", f"{model_name}_review", category)
                    for model_name in llm_models
                }

                if not os.path.exists(section_map_dir):
                    continue  # Skip if section directory does not exist

                paper_files = {
                    f.replace("_section.json", ""): os.path.join(section_map_dir, f)
                    for f in os.listdir(section_map_dir) if f.endswith("_section.json")
                }

                # Load real reviews
                real_reviews_data = []
                if os.path.exists(real_review_file):
                    real_reviews_data = load_json(real_review_file)
                    real_reviews_by_id = {entry["paper_id"]: entry["reviews"] for entry in real_reviews_data}
                else:
                    real_reviews_by_id = {}

                paper_ids = list(paper_files.keys())

                for paper_id in tqdm(paper_ids, desc=f"{conf}-{year}-{category}", ncols=100):

                    section_file = paper_files[paper_id]
                    section_embeddings = encode_sections(section_file)

                    # === Process real reviews ===
                    real_output_path = os.path.join(real_output_dir, f"{paper_id}.json")
                    if not os.path.exists(real_output_path) and paper_id in real_reviews_by_id:
                        reviews = real_reviews_by_id[paper_id]
                        similarity_reviews = process_reviews(paper_id, section_embeddings, reviews, review_source="real")
                        save_json(similarity_reviews, real_output_path)

                    # === Process LLM reviews ===
                    for model_name in llm_models:
                        llm_review_dir = llm_review_dirs[model_name]
                        llm_output_dir = llm_output_dirs[model_name]
                        review_path = os.path.join(llm_review_dir, f"{paper_id}.json")
                        llm_output_path = os.path.join(llm_output_dir, f"{paper_id}.json")

                        if not os.path.exists(llm_output_path) and os.path.exists(review_path):
                            llm_reviews_data = load_json(review_path)
                            reviews = llm_reviews_data.get("reviews", [])
                            similarity_reviews = process_reviews(paper_id, section_embeddings, reviews, review_source="llm")
                            save_json(similarity_reviews, llm_output_path)

if __name__ == "__main__":
    main()
    print("\n All similarity analysis completed!")
