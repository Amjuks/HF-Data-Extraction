# Dataset Pipeline

This repo reads Hugging Face dataset IDs, detects code-generation data, converts it to a unified format, and writes output to CSV, Parquet, or both.

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

3. Add datasets to `datasets.csv`

```csv
dataset_id
iamtarun/python_code_instructions_18k_alpaca
```

4. Run

```bash
python pipeline.py
```

Default output is CSV.

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

## Files You Get

- `combined_dataset.csv` (if CSV mode is enabled)
- `combined_dataset.parquet` (if Parquet mode is enabled)
- `pipeline_progress_log.csv` (always written)

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

`pipeline_progress_log.csv` shows what happened for each dataset:

- `started`
- `added`
- `skipped`
- `failed`

It also includes error messages when something fails.

## Optional Settings

```env
MAX_SAMPLE_ROWS=12
MAX_ROWS_PER_DATASET=
PROGRESS_LOG_EVERY=100
PARQUET_BATCH_SIZE=1000
WRITE_BATCH_SIZE=500
HF_STREAMING=true
DEFAULT_SPLIT=train
```

Notes:

- Leave `MAX_ROWS_PER_DATASET` empty to process full datasets.
- If `LLM_API_KEY` is missing, pipeline still runs with fallback schema detection.
- If `HF_API_KEY` (or `HUGGINGFACE_API_KEY` / `HF_TOKEN`) is set, it is used for Hugging Face dataset access. If not set, public datasets are loaded without auth.
