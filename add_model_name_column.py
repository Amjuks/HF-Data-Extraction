import os
import sys
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import requests

# Load environment variables
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
    meta_url = f"{HF_API_URL}{dataset_id}"
    hf_headers = {}
    if HF_API_KEY:
        hf_headers['Authorization'] = f'Bearer {HF_API_KEY}'
    try:
        meta_resp = requests.get(meta_url, headers=hf_headers, timeout=10)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
    except Exception as e:
        print(f"    Failed to fetch metadata: {e}")
        meta = {}
    readme_url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"
    try:
        readme_resp = requests.get(readme_url, headers=hf_headers, timeout=10)
        if readme_resp.status_code == 200:
            readme = readme_resp.text
        else:
            readme = ''
    except Exception as e:
        print(f"    Failed to fetch README: {e}")
        readme = ''
    return meta, readme

def query_llm_model_name(meta, readme):
    """Query the LLM to extract the closed model's name as JSON."""
    user_content = f"""Given the following Hugging Face dataset metadata and README, extract the name of the closed model that this dataset is based on. Respond ONLY with a JSON object in the format: {{\"model_name\": \"<model name>\"}}. If not found, respond with {{\"model_name\": null}}.\n\nMetadata:\n{meta}\n\nREADME:\n{readme}\n"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": LLM_REASONING_HINT},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    try:
        resp = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        result = resp.json()
        answer = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        # Try to extract JSON
        match = re.search(r'\{.*\}', answer, re.DOTALL)
        if match:
            try:
                model_json = json.loads(match.group(0))
                return model_json.get('model_name')
            except Exception:
                return None
        return None
    except Exception as e:
        print(f"    LLM query failed: {e}")
        return None

def process_labeled_csv(csv_path):
    print(f"\nProcessing file: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'model_name' not in df.columns:
        df['model_name'] = None
    # Only process rows where closed_model is True and model_name is not set
    mask = (df['closed_model'] == True) | (df['closed_model'] == 'True')
    rows = df[mask]
    for idx in tqdm(rows.index, desc=os.path.basename(csv_path)):
        dataset_id = df.at[idx, 'dataset_id'] if 'dataset_id' in df.columns else None
        if not dataset_id:
            continue
        print(f"  [{idx}] Dataset: {dataset_id}")
        meta, readme = fetch_hf_metadata_and_readme(dataset_id)
        model_name = query_llm_model_name(meta, readme)
        print(f"    model_name: {model_name}")
        df.at[idx, 'model_name'] = model_name
    df.to_csv(csv_path, index=False)
    print(f"Updated file: {csv_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Add model_name column to labeled CSVs.")
    parser.add_argument('--file', type=str, help="Path to a specific _labeled.csv file to process.")
    args = parser.parse_args()
    files = []
    if args.file:
        if os.path.isfile(args.file) and args.file.endswith('_labeled.csv'):
            files = [args.file]
        else:
            print("Provided file is not a valid _labeled.csv file.")
            sys.exit(1)
    else:
        for fname in os.listdir(DATASETS_DIR):
            if fname.endswith('_labeled.csv'):
                files.append(os.path.join(DATASETS_DIR, fname))
    if not files:
        print("No _labeled.csv files found to process.")
        return
    for f in files:
        process_labeled_csv(f)
    print("All files processed.")

if __name__ == "__main__":
    main()
