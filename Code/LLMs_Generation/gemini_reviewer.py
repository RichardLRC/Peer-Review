import os
import json
import re
import time
from google import genai
from tqdm import tqdm

assistant_instructions_iclr = """
    You are a professional academic paper reviewer. Evaluate papers based on the grading rubric provided.
    Return your response in JSON format with the following structure:
    
    {
      "title": "Paper Title",
      "review": { 
        "summary": "Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.",
        "soundness": { Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence.
                        4: excellent
                        3: good
                        2: fair
                        1: poor
                    },
        "presentation": { Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work.
                            4: excellent
                            3: good
                            2: fair
                            1: poor
                        },
        "Contribution": {  Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader ICLR community?
                            4: excellent
                            3: good
                            2: fair
                            1: poor
                        },
        "strengths": "Please provide a thorough assessment of the strengths of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage people to be broad in their definitions of originality and significance. For example, originality may arise from creative combinations of existing ideas, application to a new domain, or removing restrictive assumptions from prior theoretical results. You can incorporate Markdown and Latex into your review",
        "weaknesses": "Please provide a thorough assessment of the weaknesses of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage people to be broad in their definitions of originality and significance. For example, originality may arise from creative combinations of existing ideas, application to a new domain, or removing restrictive assumptions from prior theoretical results. You can incorporate Markdown and Latex into your review",
        "questions": "Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This can be very important for a productive rebuttal and discussion phase with the authors.",
        "overall_rating": Please provide an "overall rating" for this submission. Choices:
                            10: strong accept, should be highlighted at the conference.
                            8: accept, good paper.
                            6: marginally above the acceptance threshold.
                            5: marginally below the acceptance threshold.
                            3: reject, not good enough.
                            1: strong reject.
,
        "Reasons for overall_rating": Reasons content,
        "confidence": Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation. 
                        5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
                        4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
                        3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
                        2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
                        1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.

      }
    }
    """
    
assistant_instructions_neurips = """
    You are a professional academic paper reviewer. Evaluate papers based on the grading rubric provided.
    Return your response in JSON format with the following structure:
    
    {
      "title": "Paper Title",
      "review": { 
        "summary": "Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.",
        "soundness": { Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence.
                        4: excellent
                        3: good
                        2: fair
                        1: poor
                    },
        "presentation": { Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work.
                            4: excellent
                            3: good
                            2: fair
                            1: poor
                        },
        "Contribution": {  Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader ICLR community?
                            4: excellent
                            3: good
                            2: fair
                            1: poor
                        },
        "strengths": "Please provide a thorough assessment of the strengths of the paper, touching on each of the following dimensions: a. Originality: Are the tasks or methods new? Is the work a novel combination of well-known techniques? (This can be valuable!) Is it clear how this work differs from previous contributions? Is related work adequately cited?
                        b. Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?
                        c. Clarity: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
                        d. Significance: Are the results important? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance the state of the art in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?
                        You can incorporate Markdown and Latex into your review.", 
        "weaknesses": "Please provide a thorough assessment of the weaknesses of the paper, touching on each of the following dimensions: a. Originality: Are the tasks or methods new? Is the work a novel combination of well-known techniques? (This can be valuable!) Is it clear how this work differs from previous contributions? Is related work adequately cited?
                        b. Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?
                        c. Clarity: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
                        d. Significance: Are the results important? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance the state of the art in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?
                        You can incorporate Markdown and Latex into your review.",
        "questions": "Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This can be very important for a productive rebuttal and discussion phase with the authors.",
        "overall_rating": Please provide an "overall rating" for this submission. Choices:
                            10: Award quality: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
                            9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI and excellent impact on multiple areas of AI, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
                            8: Strong Accept: Technically strong paper with, with novel ideas, excellent impact on at least one area of AI or high-to-excellent impact on multiple areas of AI, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
                            7: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
                            6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
                            5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
                            4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
                            3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
                            2: Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations.
                            1: Very Strong Reject: For instance, a paper with trivial results or unaddressed ethical considerations.
        "Reasons for overall_rating": Reasons content,
        "confidence": Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation. 
                        5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
                        4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
                        3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
                        2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
                        1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.

      }
    }
    """




# Fix illegal escape characters in generated content
def fix_illegal_escapes(text):
    return re.sub(r'(?<!\\)\\(?![nrt"\\/bfu])', r'\\\\', text)


GEMINI_API_KEY="Your_API_Key"
client = genai.Client(api_key=GEMINI_API_KEY)


# Read markdown file content
def extract_text_from_mmd(mmd_path):
    with open(mmd_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"(##\s*References\b[\s\S]*?)(\n##\s+[^\n]+)", r"\2", content)
    return content.strip()


# Generate a single review using Gemini
def generate_single_review(conf, paper_text, max_retries=2):
    prompt = assistant_instructions_iclr if conf == "ICLR" else assistant_instructions_neurips
    full_prompt = f"{prompt.strip()}\n\nHere is the research paper for review:\n\n{paper_text}"
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=full_prompt,
            )
            content = response.text.strip()
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)

            with open("../gemini_debug_log.txt", "a", encoding="utf-8") as f:
                f.write("\n\n===== GEMINI OUTPUT START =====\n")
                f.write(content)
                f.write("\n===== GEMINI OUTPUT END =====\n")
            fixed_content = fix_illegal_escapes(content)
            review_json = json.loads(fixed_content)
            return review_json.get("review", review_json)

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(5)

    print("All Gemini attempts failed.")
    return None

# Process one conference per year
def process_one(conf, year):
    root_dir = "../Data"
    labels = ["good", "borderline", "bad"]

    for label in labels:
        mmd_folder = os.path.join(root_dir, conf, year, "markdown", label)
        review_output_folder = os.path.join(root_dir, conf, year, "gemini_review", f"{label}_papers")
        real_review_file = os.path.join(root_dir, conf, year, "real_review", f"{label}_papers", f"{label}_reviews.json")


        if not os.path.exists(real_review_file):
            print(f"Skipped: Missing real review file {real_review_file}")
            continue

        os.makedirs(review_output_folder, exist_ok=True)

        with open(real_review_file, "r", encoding="utf-8") as f:
            real_reviews = json.load(f)

        paper_review_counts = {}
        paper_titles = {}
        for p in real_reviews:
            paper_id = p.get("paper_id") or p.get("id")
            reviews = p.get("reviews") or []
            if paper_id:
                paper_review_counts[paper_id] = len(reviews)
                paper_titles[paper_id] = p.get("title", "Unknown Title")

        mmd_files = [f for f in os.listdir(mmd_folder) if f.endswith(".mmd")]
        for mmd_file in tqdm(mmd_files, desc=f"📂 {conf}{year} - {label}"):
            paper_id = mmd_file.replace(".mmd", "")
            output_path = os.path.join(review_output_folder, f"{paper_id}.json")

            if os.path.exists(output_path):
                continue

            paper_text = extract_text_from_mmd(os.path.join(mmd_folder, mmd_file))
            num_reviews = paper_review_counts.get(paper_id, 1)
            title = paper_titles.get(paper_id, "Unknown Title")

            print(f"Generating reviews for {paper_id} ({num_reviews} total)")

            gpt_reviews = []
            max_total_attempts = num_reviews * 3
            attempts = 0

            while len(gpt_reviews) < num_reviews and attempts < max_total_attempts:
                review = generate_single_review(conf, paper_text)
                attempts += 1
                if review:
                    gpt_reviews.append(review)
                else:
                    print(f"Attempt {attempts} failed (target {num_reviews}, success {len(gpt_reviews)})")

            if len(gpt_reviews) == num_reviews:
                result = {
                    "paper_id": paper_id,
                    "title": title,
                    "reviews": gpt_reviews
                }
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Saved successfully: {output_path}")
            else:
                print(f"Final incomplete: {paper_id} ({len(gpt_reviews)}/{num_reviews} successful)")


# Run all conference-year combinations
def process_all():
    for conf, years in {
        "ICLR": ["2024", "2025"],
        "NeurIPS": ["2023", "2024"]
    }.items():
        for year in years:
            process_one(conf, year)
if __name__ == "__main__":
    # process_all()
    process_one("ICLR", "2025")  
    process_one("NeurIPS", "2023")
    process_one("NeurIPS", "2024")
    process_one("ICLR", "2024")

