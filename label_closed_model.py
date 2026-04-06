import os
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME')
LLM_API_KEY = os.getenv('LLM_API_KEY')
LLM_BASE_URL = os.getenv('LLM_BASE_URL')
LLM_USE_CHAT_COMPLETIONS = os.getenv('LLM_USE_CHAT_COMPLETIONS', 'true').lower() == 'true'
LLM_REASONING_HINT = os.getenv('LLM_REASONING_HINT', 'Reasoning: Low')

HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {LLM_API_KEY}'
}

DATASETS_DIR = os.path.join(os.path.dirname(__file__), 'datasets')


HF_API_URL = 'https://huggingface.co/api/datasets/'
HF_API_KEY = os.getenv('HF_API_KEY')


def fetch_hf_metadata_and_readme(dataset_id):
    """Fetch metadata and README for a Hugging Face dataset."""
    meta_url = f"{HF_API_URL}{dataset_id}"
    hf_headers = {}
    if HF_API_KEY:
        hf_headers['Authorization'] = f'Bearer {HF_API_KEY}'
    print(f"  Fetching metadata for: {dataset_id}")
    try:
        meta_resp = requests.get(meta_url, headers=hf_headers, timeout=10)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        print(f"    Metadata fetched.")
    except Exception as e:
        print(f"    Failed to fetch metadata: {e}")
        meta = {}
    # Try to fetch README
    readme_url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"
    print(f"  Fetching README for: {dataset_id}")
    try:
        readme_resp = requests.get(readme_url, headers=hf_headers, timeout=10)
        if readme_resp.status_code == 200:
            readme = readme_resp.text
            print(f"    README fetched.")
        else:
            readme = ''
            print(f"    README not found (status {readme_resp.status_code}).")
    except Exception as e:
        print(f"    Failed to fetch README: {e}")
        readme = ''
    return meta, readme


def query_llm_is_closed_model(meta, readme):
    """Query the LLM to determine if the dataset is from a closed model."""
    user_content = f"""Given the following Hugging Face dataset metadata and README, is this dataset from a closed model? Answer only true or false.\n\nMetadata:\n{meta}\n\nREADME:\n{readme}\n"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": LLM_REASONING_HINT},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    print("  Querying LLM for closed_model label...")
    try:
        resp = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        result = resp.json()
        # Try to extract the answer
        answer = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip().lower()
        print(f"    LLM response: {answer}")
        if 'true' in answer:
            return True
        elif 'false' in answer:
            return False
        else:
            print("    LLM response unclear, labeling as None.")
            return None
    except Exception as e:
        print(f"    LLM query failed: {e}")
        return None


def process_csv_file(csv_path):
    print(f"\nProcessing file: {csv_path}")
    df = pd.read_csv(csv_path)
    out_path = os.path.splitext(csv_path)[0] + '_labeled.csv'
    # Open output file and write header
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=['dataset_id', 'link', 'closed_model'])
        writer.writeheader()
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(csv_path)):
            dataset_id = row.get('dataset_id')
            link = row.get('link')
            print(f"\n[{idx+1}/{len(df)}] Dataset: {dataset_id}")
            if not dataset_id or not link:
                print("  Skipping row: missing dataset_id or link.")
                continue
            meta, readme = fetch_hf_metadata_and_readme(dataset_id)
            closed_model = query_llm_is_closed_model(meta, readme)
            print(f"  closed_model label: {closed_model}")
            writer.writerow({
                'dataset_id': dataset_id,
                'link': link,
                'closed_model': closed_model
            })
    print(f"\nWrote labeled file: {out_path}\n")


def main():
    print(f"Scanning directory: {DATASETS_DIR}\n")
    for fname in os.listdir(DATASETS_DIR):
        if fname.endswith('.csv'):
            process_csv_file(os.path.join(DATASETS_DIR, fname))
    print("\nAll files processed.")

if __name__ == "__main__":
    main()
