# Dataset Pipeline

This repo reads Hugging Face datasets, checks whether they match a target domain, converts them to a unified conversation schema, and writes output to CSV, Parquet, or both.

It now supports:

- content-based schema detection for `code_generation`, `math`, and `natural_language`
- nested conversation fields such as `conversations`, `messages`, or chat payloads stored inside JSON objects
- parallel processing of multiple domains while throttling LLM schema-analysis calls
- Hugging Face API throttling for config/split/dataset loading
- a required `language` field in the normalized schema

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Create `.env`

```env
LLM_API_KEY=your_key_here
MODEL_NAME=gpt-4o-mini
HF_API_KEY=your_huggingface_token_optional

OUTPUT_CSV_FILE=combined_dataset.csv
OUTPUT_PARQUET_FILE=combined_dataset.parquet
RUN_LOG_FILE=pipeline_progress_log.csv
```

3. Create one CSV per domain. Each CSV must contain a `link` column with a Hugging Face dataset URL.

Example `code_generation.csv`:

```csv
link
https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca
```

4. Configure domains in `domains.json`

```json
{
  "domains": [
    {
      "name": "code_generation",
      "task_type": "code_generation",
      "csv_path": "code_generation.csv"
    },
    {
      "name": "math",
      "task_type": "math",
      "csv_path": "math.csv"
    },
    {
      "name": "natural_language",
      "task_type": "natural_language",
      "csv_path": "natural_language.csv"
    }
  ]
}
```

5. Run

```bash
python pipeline.py
```

Default output is CSV. Each domain writes its own output files, for example `combined_dataset_code_generation.csv` and `combined_dataset_math.csv`.

Normalized output fields:

- `conversation`
- `language`
- `reasoning`
- `metadata`
- `dataset_id`

`language` is always normalized to lowercase English labels such as `english`, `hindi`, `spanish`, `multilingual`, or `unknown`.

All available splits for a dataset are processed together. The pipeline does not limit itself to `train` or `test`.

## Output Options

CSV only:

```bash
python pipeline.py --output-format csv
```

Parquet only:

```bash
python pipeline.py --output-format parquet
```

Both CSV and Parquet:

```bash
python pipeline.py --output-format both
```

Use a custom domain config file:

```bash
python pipeline.py --domains-config custom_domains.json
```

Process only specific domains from the config:

```bash
python pipeline.py --domains code_generation
```

Multiple domains:

```bash
python pipeline.py --domains code_generation math
```

Resume from previous progress (default progress name):

```bash
python pipeline.py --resume
```

Resume from a named progress:

```bash
python pipeline.py --resume myrun
```

Start a new named progress (without resume):

```bash
python pipeline.py --progress-name myrun
```

## Files You Get

- `combined_dataset_<domain>.csv` (if CSV mode is enabled)
- `combined_dataset_<domain>.parquet` (if Parquet mode is enabled)
- `pipeline_progress_log_<domain>.csv` (always written)
- `.pipeline_checkpoint_<domain>_<progress>.json`
- `.dataset_registry.json` for persistent dataset-level skip tracking across runs
- `pipeline_summary_<progress>.json` with direct numeric totals by domain and overall

## Deduplicate Any CSV/Parquet

Use:

```bash
python deduplicate.py --input your_file.csv
```

This creates `your_file_dedup.csv`.

Parquet example:

```bash
python deduplicate.py --input your_file.parquet
```

Keep only unique prompt/response pairs:

```bash
python deduplicate.py --input combined_dataset.csv --subset conversation,reasoning --keep first
```

Case-insensitive dedup on selected columns:

```bash
python deduplicate.py --input combined_dataset.csv --subset conversation --ignore-case --strip-whitespace
```

Write to a custom file:

```bash
python deduplicate.py --input combined_dataset.parquet --output combined_dataset_dedup.parquet
```

## Progress Log (Important)

Each domain log shows what happened for each dataset:

- `started`
- `added`
- `skipped`
- `deferred`
- `failed`

It also includes error messages when something fails.

For named progress runs, files are automatically suffixed, for example:

- `combined_dataset_code_generation_myrun.csv`
- `combined_dataset_math_myrun.parquet`
- `pipeline_progress_log_natural_language_myrun.csv`
- `.pipeline_checkpoint_code_generation_myrun.json`

## Optional Settings

```env
MAX_SAMPLE_ROWS=12
MAX_ROWS_PER_DATASET=
PROGRESS_LOG_EVERY=100
PARQUET_BATCH_SIZE=1000
WRITE_BATCH_SIZE=500
HF_STREAMING=true
DOMAINS_CONFIG_PATH=domains.json
MAX_PARALLEL_DOMAINS=3
LLM_MAX_CONCURRENCY=2
LLM_MIN_INTERVAL_SECONDS=0.0
HF_MAX_CONCURRENCY=2
HF_MIN_INTERVAL_SECONDS=0.2
```

Notes:

- Leave `MAX_ROWS_PER_DATASET` empty to process full datasets.
- `MAX_ROWS_PER_DATASET` applies across the combined stream of all splits for a dataset.
- A domain entry can override `max_rows_per_dataset` for just that domain.
- A domain entry can use `csv_path` with a `link` column, or `dataset_ids` for direct IDs.
- Use `--domains ...` to process only selected configured domains without editing `domains.json`.
- Dataset links are normalized to canonical Hugging Face dataset IDs before processing.
- If `LLM_API_KEY` is missing, pipeline still runs with fallback schema detection.
- If `HF_API_KEY` (or `HUGGINGFACE_API_KEY` / `HF_TOKEN`) is set, it is used for Hugging Face dataset access. If not set, public datasets are loaded without auth.
- API rate limits are retried automatically with exponential backoff for both OpenAI schema calls and Hugging Face dataset API calls.
- When a provider supplies `Retry-After`, the retry logic waits at least that long before retrying.
- Parallel domain execution is bounded by `MAX_PARALLEL_DOMAINS`, and LLM schema analysis is further gated by `LLM_MAX_CONCURRENCY` and `LLM_MIN_INTERVAL_SECONDS`.
- Hugging Face API access is further gated by `HF_MAX_CONCURRENCY` and `HF_MIN_INTERVAL_SECONDS` to stay under limits during parallel runs.
- If Hugging Face temporarily requires authentication after IP throttling, or the local `datasets` package cannot parse a dataset feature schema, the dataset is marked `deferred` instead of `failed` so later reruns can retry it cleanly.
- Main output is deduplicated by default (`conversation + reasoning`). To disable: `--no-deduplicate`.
- Processed datasets are also tracked in `.dataset_registry.json`, so reruns skip the same domain/dataset pair unless it previously failed.
- If `domains.json` is missing, the pipeline falls back to the legacy `datasets.csv` flow and treats those datasets as `code_generation`.
