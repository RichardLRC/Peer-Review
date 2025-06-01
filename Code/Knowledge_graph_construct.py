import os
import json
import csv
import networkx as nx
from collections import Counter
import math
import random
# ===== Helper function to build entity graph from predictions =====
def build_entity_graph(predicted_ner, predicted_re, flat_sentences):
    G = nx.DiGraph()
    undirected_relations = {'COMPARE', 'CONJUNCTION'}

    for ent in predicted_ner:
        start, end, label = ent
        try:
            token = " ".join(flat_sentences[start:end+1])
        except IndexError:
            token = ""
        G.add_node((start, end), label=label, text=token)

    for rel in predicted_re:
        h_span, t_span, rel_type = rel
        h = (h_span[0], h_span[1])
        t = (t_span[0], t_span[1])
        if h in G.nodes and t in G.nodes:
            G.add_edge(h, t, relation=rel_type)
            if rel_type in undirected_relations:
                G.add_edge(t, h, relation=rel_type)

    return G

# ===== Graph metrics =====
def compute_graph_metrics(G):
    metrics = {}
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['avg_degree'] = sum(dict(G.degree()).values()) / max(1, G.number_of_nodes())

    labels = [G.nodes[n]['label'] for n in G.nodes if 'label' in G.nodes[n]]
    label_count = Counter(labels)
    total_labels = sum(label_count.values())
    if total_labels > 0:
        metrics['label_entropy'] = -sum((c / total_labels) * math.log(c / total_labels, 2) for c in label_count.values())
    else:
        metrics['label_entropy'] = 0.0

    return metrics


# ===== Determine if a real review question section is empty/invalid =====
def should_filter_question(example):
    return example['doc_key'].endswith('_questions') and len(example['sentences']) == 1 and len(example['sentences'][0]) < 10

# ===== Load graphs with aligned filtering =====
def load_graphs_multi_source(base_dir, sources, conference, year, category):
    graphs = {}
    skip_count = {}
    source_lines = {src: [] for src in sources}

    # Step 1: Collect real review skip counts
    for source in sources:
        jsonl_path = os.path.join(base_dir, source, conference, year, f"merged_ent_pred_with_rel_{category}.jsonl")
        if not os.path.exists(jsonl_path):
            continue

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]

        if source == 'real':
            for line in lines:
                if should_filter_question(line):
                    paper_id = line['doc_key'].split('_')[0]
                    skip_count[paper_id] = skip_count.get(paper_id, 0) + 1

        source_lines[source] = lines

    # Step 2: Build graphs with filtering
    for source in sources:
        paper_question_map = {}

        for line_idx, example in enumerate(source_lines[source]):
            doc_key = example['doc_key']
            paper_id = doc_key.split('_')[0]
            section = doc_key.split('_')[-1]
            key = (paper_id, section, source, line_idx)

            if source == 'real' and should_filter_question(example):
                continue

            if source != 'real' and section == 'questions':
                paper_question_map.setdefault(paper_id, []).append((key, example))
            else:
                flat_tokens = [tok for sent in example['sentences'] for tok in sent]
                all_ner = [(s, e, l) for ner in example['predicted_ner'] for (s, e, l) in ner]
                all_re = [(h, t, r) for (s_idx, rels) in example['predicted_re'] for (h, t, r) in rels]
                G = build_entity_graph(all_ner, all_re, flat_tokens)
                graphs[key] = G

        # Randomly skip questions in LLMs to match real
        if source != 'real':
            for paper_id, items in paper_question_map.items():
                to_skip = skip_count.get(paper_id, 0)
                selected = random.sample(items, k=max(0, len(items)-to_skip)) if to_skip < len(items) else []
                for key, example in selected:
                    flat_tokens = [tok for sent in example['sentences'] for tok in sent]
                    all_ner = [(s, e, l) for ner in example['predicted_ner'] for (s, e, l) in ner]
                    all_re = [(h, t, r) for (s_idx, rels) in example['predicted_re'] for (h, t, r) in rels]
                    G = build_entity_graph(all_ner, all_re, flat_tokens)
                    graphs[key] = G

    return graphs

# ===== Save metrics to CSV =====
def save_metrics_to_csv(graphs, output_path):
    if not graphs:
        return

    sample_metrics = compute_graph_metrics(next(iter(graphs.values())))
    fieldnames = ['paper_id', 'section', 'source_type', 'line_index'] + list(sample_metrics.keys())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for key, G in graphs.items():
            paper_id, section, source_type, line_index = key
            metrics = compute_graph_metrics(G)
            row = {'paper_id': paper_id, 'section': section, 'source_type': source_type, 'line_index': line_index}
            row.update(metrics)
            writer.writerow(row)

# ===== Main Execution =====
def main():
    base_dir = '../Data/Knowledge_Graph/'
    output_base = '../Data/Knowledge_Graph/'
    conferences_years = {
        "ICLR": ["2024", "2025"],
        "NeurIPS": ["2023", "2024"]
    }
    categories = ["good", "borderline", "bad"]
    sources = ["claude", "gemini", "gpt", "llama", "qwen", "real"]

    for conf, years in conferences_years.items():
        for year in years:
            for category in categories:
                print(f"\U0001F680 Processing {conf} {year} {category}...")
                graphs = load_graphs_multi_source(base_dir, sources, conf, year, category)
                output_csv = os.path.join(output_base, conf, year, category, "graph_metrics_clean.csv")
                save_metrics_to_csv(graphs, output_csv)

    print("\n✅ All graph metrics extraction completed!")

if __name__ == "__main__":
    main()
