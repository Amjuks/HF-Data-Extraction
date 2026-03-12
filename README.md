# Dataset Pipeline

Production-ready Python pipeline to discover, classify, and normalize code-generation datasets from Hugging Face into a unified CSV format.

## Features

- Reads dataset IDs from `datasets.csv`
- Detects dataset configs and splits
- Loads Hugging Face datasets with streaming support
- Samples rows and infers schema using an LLM
- Classifies irrelevant datasets and skips them safely
- Converts supported dataset shapes into a unified conversation schema
- Writes output incrementally to `combined_dataset.csv`
- Includes retries, validation, and robust fallbacks

## Project Structure

```text
.
  README.md
  requirements.txt
  config.py
  datasets.csv
  pipeline.py
  modules/
    __init__.py
    csv_reader.py
    csv_writer.py
    dataset_loader.py
    sample_extractor.py
    schema_agent.py
    dataset_classifier.py
    dataset_converter.py
    utils.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables in a `.env` file:

```bash
MAX_SAMPLE_ROWS=12
MAX_ROWS_PER_DATASET=
OUTPUT_FILE=combined_dataset.csv
MODEL_NAME=gpt-4o-mini
LLM_API_KEY=your_key_here
PROGRESS_LOG_EVERY=100
FLUSH_EACH_RECORD=true
# Optional:
# LLM_BASE_URL=https://your-openai-compatible-endpoint
# HF_STREAMING=true
# DEFAULT_SPLIT=train
```

4. Edit `datasets.csv`:

```csv
dataset_id
iamtarun/python_code_instructions_18k_alpaca
```

## Run

```bash
python pipeline.py
```

Output is written to:

`combined_dataset.csv`

## Notes

- If `LLM_API_KEY` is not set or schema inference fails, the pipeline falls back to heuristic schema detection.
- By default `MAX_ROWS_PER_DATASET` is unlimited (entire dataset is processed). Set it to a number only when you want to cap.
- Output is appended incrementally and flushed per record (configurable), so `combined_dataset.csv` updates in real time.
- In CSV, `conversation` and `metadata` are JSON-encoded strings.

