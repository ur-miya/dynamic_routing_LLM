import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("[WARNING] sentence-transformers not installed. Clustering will be skipped.")


JUDGE_PROMPT_TEMPLATE = """You are a strict NLG evaluator. Analyze the student model's response quality.

PROMPT (user request):
{prompt}

REFERENCE ANSWER (human reply from dataset):
{reference}

STUDENT ANSWER:
{student_response}

Task: Identify the SINGLE most important and clearly distinguishable problem in the student answer compared to the reference.
Describe it in 1-2 sentences. Be specific and concise.
Focus on the main issue only (e.g., "too short and missing key information", "factually incorrect claim about X",
"does not follow the requested format", "off-topic response", "repetitive/incoherent text").

Problem description:"""


def call_teacher_api(
    prompt_text: str,
    url_override: str = None,
    model_override: str = None,
    token_override: str = None,
    api_path_override: str = None,
    max_tokens: int = 150,
    no_think: bool = True,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    ) -> str:

    teacher_url   = url_override or os.getenv("TEACHER_URL", "")
    api_path      = api_path_override or os.getenv("TEACHER_API_PATH", "/v1/completions")
    teacher_model = model_override or os.getenv("TEACHER_MODEL", "")
    teacher_token = token_override or os.getenv("TEACHER_TOKEN", "")

    if not teacher_url or not teacher_model:
        return "ERROR: TEACHER_URL or TEACHER_MODEL not set"

    if no_think:
        prompt_text = "/no_think\n" + prompt_text

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {teacher_token}",
    }
    payload = {
        "model": teacher_model,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    url = teacher_url.rstrip("/") + api_path

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["text"].strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            return text

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARNING] API call failed (attempt {attempt+1}): {e}. Retrying...")
                time.sleep(retry_delay)
            else:
                print(f"[ERROR] API call failed after {max_retries} attempts: {e}")
                return "ERROR: API call failed"

    return "ERROR: max retries exceeded"


class IncrementalErrorClusterer:

    def __init__(self, embedder, threshold: float = 0.75):
        self.embedder = embedder
        self.threshold = threshold
        self.clusters = []     

    def add(self, description: str) -> int:

        emb = self.embedder.encode([description], normalize_embeddings=True)[0]

        if not self.clusters:
            return self._new_cluster(description, emb)

        centroids = np.array([c["centroid"] for c in self.clusters])
        sims = cosine_similarity([emb], centroids)[0]
        best_idx = int(np.argmax(sims))

        if sims[best_idx] >= self.threshold:
            self.clusters[best_idx]["descriptions"].append(description)
            self.clusters[best_idx]["members"] += 1
            n = self.clusters[best_idx]["members"]
            self.clusters[best_idx]["centroid"] = (
                (self.clusters[best_idx]["centroid"] * (n - 1) + emb) / n
            )
            return best_idx
        else:
            return self._new_cluster(description, emb)

    def _new_cluster(self, description: str, emb: np.ndarray) -> int:
        cluster_id = len(self.clusters)
        self.clusters.append({
            "cluster_id": cluster_id,
            "name": f"cluster_{cluster_id}",   # будет переименован вручную или авто
            "centroid": emb,
            "members": 1,
            "descriptions": [description],
        })
        return cluster_id

    def get_taxonomy(self) -> pd.DataFrame:
        rows = []
        for c in self.clusters:
            rows.append({
                "cluster_id": c["cluster_id"],
                "cluster_name": c["name"],
                "count": c["members"],
                "example_descriptions": " | ".join(c["descriptions"][:3]),
            })
        df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
        return df

    def auto_name_clusters(self, top_n: int = 3):

        for c in self.clusters:
            first_desc = c["descriptions"][0]
            words = first_desc.split()[:5]
            c["name"] = " ".join(words).lower().rstrip(".,:")


def main():
    parser = argparse.ArgumentParser(
        description='Error profiling via LLM-as-a-qualitative-judge + clustering'
    )
    parser.add_argument(
        '--features_csv', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing/features_er.csv'
        ),
        help='Path to features_er.csv'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs/routing'
        ),
    )
    parser.add_argument('--max_error_samples', type=int, default=1000,
                        help='Max number of error samples to analyze (label=1)')
    parser.add_argument('--clustering_threshold', type=float, default=0.75,
                        help='Cosine similarity threshold for clustering (0-1)')
    parser.add_argument('--embed_model', type=str,
                        default='sentence-transformers/all-mpnet-base-v2',
                        help='SentenceTransformer model for clustering embeddings')
    parser.add_argument('--resume', action='store_true',
                        help='Resume: skip prompts already in error_profiles_er.csv')
    parser.add_argument('--skip_api', action='store_true',
                        help='Skip API calls (only re-run clustering on existing descriptions)')
    parser.add_argument('--llm_judge_url', type=str, default=None,
                        help='Override TEACHER_URL from .env')
    parser.add_argument('--llm_judge_model', type=str, default=None,
                        help='Override TEACHER_MODEL from .env')
    parser.add_argument('--llm_judge_token', type=str, default=None,
                        help='Override TEACHER_TOKEN from .env')
    parser.add_argument('--llm_judge_api_path', type=str, default=None,
                        help='Override TEACHER_API_PATH from .env ')
    parser.add_argument('--llm_judge_max_tokens', type=int, default=150,
                        help='Max tokens for LLM-judge response')
    parser.add_argument('--llm_judge_batch_delay', type=float, default=0.5,
                        help='Delay between API calls in seconds (avoid rate limiting)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for embeddings')
    parser.add_argument('--llm_judge_no_think', action='store_true',
                    help='Add /no_think to teacher prompt (for vLLM completions API)')

    args = parser.parse_args()

    if not os.path.exists(args.features_csv):
        print(f"Error: features CSV not found: {args.features_csv}")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    output_profiles = os.path.join(args.output_dir, "error_profiles_er.csv")
    output_taxonomy = os.path.join(args.output_dir, "error_taxonomy_er.csv")

    print(f"Loading {args.features_csv}")
    df = pd.read_csv(args.features_csv)
    df_errors = df[df["binary_label"] == 1].copy().reset_index(drop=True)
    print(f"Total samples: {len(df)}")
    print(f"Error samples (label=1): {len(df_errors)}")

    if len(df_errors) > args.max_error_samples:
        df_errors = df_errors.head(args.max_error_samples)
        print(f"Limiting to {args.max_error_samples} error samples")

    processed_prompts = set()
    existing_records = []
    if args.resume and os.path.exists(output_profiles):
        df_existing = pd.read_csv(output_profiles)
        processed_prompts = set(df_existing["prompt"].tolist())
        existing_records = df_existing.to_dict("records")
        print(f"Resuming: {len(processed_prompts)} already processed")

    new_records = []

    if not args.skip_api:
        print(f"\n=== Per-instance error analysis ===")
        print(f"Calling teacher API (Qwen3-32B) for each error sample...")

        for _, row in tqdm(df_errors.iterrows(), total=len(df_errors), desc="LLM-judge"):
            prompt = str(row["prompt"])
            if prompt in processed_prompts:
                continue

            judge_input = JUDGE_PROMPT_TEMPLATE.format(
                prompt=prompt[:1000],                         
                reference=str(row["reply"])[:800],
                student_response=str(row["student_response"])[:800],
            )

            error_description = call_teacher_api(
                judge_input,
                url_override=args.llm_judge_url,
                model_override=args.llm_judge_model,
                token_override=args.llm_judge_token,
                api_path_override=args.llm_judge_api_path,
                max_tokens=args.llm_judge_max_tokens,
                no_think=args.llm_judge_no_think,
            )
            if args.llm_judge_batch_delay > 0:
                time.sleep(args.llm_judge_batch_delay)

            record = {
                "prompt": prompt,
                "reply": row["reply"],
                "student_response": row["student_response"],
                "rouge1": row.get("rouge1", None),
                "bert_f1": row.get("bert_f1", None),
                "binary_label": 1,
                "error_description": error_description,
                "cluster_id": -1,       
                "cluster_name": "",
            }
            new_records.append(record)

    all_records = existing_records + new_records

    df_profiles = pd.DataFrame(all_records)
    df_profiles.to_csv(output_profiles, index=False)
    print(f"\nError descriptions saved to {output_profiles} ({len(df_profiles)} records)")

    if not HAS_SBERT:
        print("[WARNING] sentence-transformers not installed, skipping clustering")
        return

    print(f"\nIncremental error clustering")
    print(f"Threshold: {args.clustering_threshold}")
    print(f"Loading embedder: {args.embed_model}")

    embedder = SentenceTransformer(args.embed_model, device=args.device)
    clusterer = IncrementalErrorClusterer(embedder, threshold=args.clustering_threshold)

    valid_mask = df_profiles["error_description"].notna() & \
                 ~df_profiles["error_description"].str.startswith("ERROR")
    df_valid = df_profiles[valid_mask].copy().reset_index(drop=True)
    df_invalid = df_profiles[~valid_mask].copy()

    print(f"Valid descriptions: {len(df_valid)} / {len(df_profiles)}")

    cluster_ids = []
    for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Clustering"):
        cid = clusterer.add(str(row["error_description"]))
        cluster_ids.append(cid)

    df_valid["cluster_id"] = cluster_ids

    clusterer.auto_name_clusters()
    id_to_name = {c["cluster_id"]: c["name"] for c in clusterer.clusters}
    df_valid["cluster_name"] = df_valid["cluster_id"].map(id_to_name)

    df_invalid["cluster_id"] = -1
    df_invalid["cluster_name"] = "error_or_missing"
    df_final = pd.concat([df_valid, df_invalid], ignore_index=True)
    df_final.to_csv(output_profiles, index=False)
    print(f"\nError profiles with clusters saved to {output_profiles}")

    taxonomy_df = clusterer.get_taxonomy()
    taxonomy_df.to_csv(output_taxonomy, index=False)
    del embedder
    import torch
    if args.device != "cpu":
        torch.cuda.empty_cache()
    print(f"Error taxonomy saved to {output_taxonomy}")

    #print(f"\nErroe taxonomy")
    #print(f"Total clusters: {len(clusterer.clusters)}")
    #print(taxonomy_df.to_string(index=False))


if __name__ == "__main__":
    main()