#!/usr/bin/env python3
"""Validate conversational pipeline output."""

import csv
import json
from pathlib import Path

# Increase CSV field limit
csv.field_size_limit(int(1e8))

output_dir = Path('output/code')
categories = ['closed_single', 'closed_multi', 'open_single', 'open_multi']

print('📊 OUTPUT FILE VALIDATION\n')
total_records = 0

for category in categories:
    filepath = output_dir / f'{category}_turn.csv'
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f'{category}_turn.csv:')
            print(f'  Records: {len(rows)}')
            total_records += len(rows)
            if rows:
                print(f'  Columns: {len(rows[0])}')
                sample = rows[0]
                print(f'  Sample: category={sample["category"]}, domain={sample["domain"]}, turn_type={sample["turn_type"]}, made_by={sample["made_by"]}')
            print()

print(f'✅ Total records processed: {total_records}')

# Check progress file
progress_file = output_dir / 'progress.json'
if progress_file.exists():
    with open(progress_file, 'r') as f:
        prog = json.load(f)
        print(f'✅ Progress file: last_processed_row={prog["last_processed_row"]}')
        print(f'   Input file: {prog["input_file"]}')
