#!/usr/bin/env python3
"""
Conversational Dataset Processing Pipeline

Processes conversational dataset CSV files and splits them into categorized outputs
based on model type (Closed vs Open) and conversation type (Single-turn vs Multi-turn).

Features:
- Row-by-row streaming processing with minimal memory usage
- Real-time output writing (append mode)
- Crash recovery using progress file
- Proper error handling for invalid data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Increase CSV field limit to handle large conversation JSON
csv.field_size_limit(int(1e8))


@dataclass
class ProgressState:
    """Tracks pipeline progress for crash recovery."""
    input_file: str
    last_processed_row: int


class ConversationalPipeline:
    """Main pipeline for processing conversational datasets."""
    
    OUTPUT_HEADERS = ["category", "messages", "difficulty", "task", "domain", "language", "source", "made_by", "turn_type"]
    CATEGORIES = ["closed_single", "closed_multi", "open_single", "open_multi"]
    
    def __init__(self, input_file: str, domain: str):
        """
        Initialize the pipeline.
        
        Args:
            input_file: Path to input CSV file
            domain: Domain name (used for output directory and CSV filename)
        """
        self.input_file = input_file
        self.domain = domain
        self.output_dir = Path("output") / domain
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "progress.json"
        
        # Output file paths
        self.output_files = {
            category: self.output_dir / f"{category}_turn.csv"
            for category in self.CATEGORIES
        }
        
        # Model lookup table (dataset_id -> model info)
        self.model_lookup = {}
        
        # Statistics
        self.stats = {
            "total_rows": 0,
            "processed_rows": 0,
            "skipped_rows": 0,
            "skipped_reasons": {}
        }
    
    def load_labeled_dataset(self) -> None:
        """
        Load the labeled dataset for the domain.
        
        Expected path: datasets/<domain>_labeled.csv
        Expected columns: dataset_id, closed_model, model_name
        """
        labeled_path = Path("datasets") / f"{self.domain}_labeled.csv"
        
        if not labeled_path.exists():
            print(f"[WARNING] Labeled dataset not found: {labeled_path}")
            return
        
        try:
            with open(labeled_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset_id = row.get('dataset_id', '').strip()
                    closed_model = row.get('closed_model', '').strip().lower() == 'true'
                    model_name = row.get('model_name', '').strip() if closed_model else ''
                    
                    if dataset_id:
                        self.model_lookup[dataset_id] = {
                            'closed_model': closed_model,
                            'model_name': model_name
                        }
            
            print(f"[OK] Loaded {len(self.model_lookup)} labeled entries from {labeled_path}")
        except Exception as e:
            print(f"[ERROR] Error loading labeled dataset: {e}")
    
    def load_progress(self) -> int:
        """
        Load progress from previous run.
        
        Returns:
            Row number to resume from (0 if starting fresh)
        """
        if not self.progress_file.exists():
            return 0
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                # Resume from the NEXT row after the last processed row
                last_processed = data.get('last_processed_row', -1)
                return last_processed + 1
        except Exception as e:
            print(f"[WARNING] Error loading progress file: {e}")
            return 0
    
    def save_progress(self, row_number: int) -> None:
        """Save current progress to file."""
        try:
            progress = ProgressState(
                input_file=self.input_file,
                last_processed_row=row_number
            )
            with open(self.progress_file, 'w') as f:
                json.dump(progress.__dict__, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Error saving progress: {e}")
    
    def initialize_output_files(self) -> dict:
        """
        Initialize output files with headers if they don't exist.
        
        Returns:
            Dict of file handles opened in append mode
        """
        file_handles = {}
        
        for category in self.CATEGORIES:
            file_path = self.output_files[category]
            is_new = not file_path.exists()
            
            # Open in append mode
            handle = open(file_path, 'a', newline='', encoding='utf-8')
            file_handles[category] = handle
            
            # Write header if file is new
            if is_new:
                writer = csv.DictWriter(handle, fieldnames=self.OUTPUT_HEADERS)
                writer.writeheader()
                handle.flush()
        
        return file_handles
    
    def classify_turn_type(self, conversation: list[dict]) -> str:
        """
        Classify conversation as single or multi-turn.
        
        Args:
            conversation: List of message dicts with 'role' and 'content'
            
        Returns:
            "single" or "multi"
        """
        return "single" if len(conversation) == 2 else "multi"
    
    def classify_model_type(self, dataset_id: str) -> tuple[str, str]:
        """
        Classify model type and determine made_by value.
        
        Args:
            dataset_id: Dataset identifier to look up
            
        Returns:
            Tuple of (model_type, made_by)
            model_type: "closed" or "open"
            made_by: model_name or "open_source"
        """
        model_info = self.model_lookup.get(dataset_id, {})
        
        if model_info.get('closed_model', False):
            return "closed", model_info.get('model_name', 'unknown')
        else:
            return "open", "open_source"
    
    def extract_task(self, metadata_str: str) -> str:
        """
        Extract task_type from metadata JSON.
        
        Args:
            metadata_str: JSON string containing metadata
            
        Returns:
            Task type or empty string if not found
        """
        try:
            metadata = json.loads(metadata_str)
            return metadata.get('task_type', '')
        except (json.JSONDecodeError, TypeError):
            return ''
    
    def process_row(
        self,
        row: dict,
        file_handles: dict,
        writers: dict
    ) -> bool:
        """
        Process a single row and write to appropriate output file.
        
        Args:
            row: CSV row dictionary
            file_handles: Dict of open file handles
            writers: Dict of CSV writers
            
        Returns:
            True if row was processed, False if skipped
        """
        try:
            # Parse conversation JSON
            try:
                conversation = json.loads(row.get('conversation', '[]'))
            except (json.JSONDecodeError, TypeError):
                self._record_skip('invalid_conversation_json')
                return False
            
            # Skip empty conversations
            if not conversation:
                self._record_skip('empty_conversation')
                return False
            
            # Parse metadata JSON
            metadata_str = row.get('metadata', '{}')
            try:
                metadata = json.loads(metadata_str)
            except (json.JSONDecodeError, TypeError):
                self._record_skip('invalid_metadata_json')
                return False
            
            # Extract fields
            dataset_id = row.get('dataset_id', '').strip()
            language = row.get('language', '').strip().lower() or 'unknown'
            
            # Classify
            turn_type = self.classify_turn_type(conversation)
            model_type, made_by = self.classify_model_type(dataset_id)
            category = f"{model_type}_{turn_type}"
            task = self.extract_task(metadata_str)
            
            # Build output row
            output_row = {
                'category': category,
                'messages': json.dumps(conversation),
                'difficulty': '',
                'task': task,
                'domain': self.domain,
                'language': language,
                'source': 'huggingface',
                'made_by': made_by,
                'turn_type': turn_type
            }
            
            # Write to appropriate file
            writer = writers[category]
            writer.writerow(output_row)
            file_handles[category].flush()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error processing row: {e}")
            self._record_skip('processing_error')
            return False
    
    def _record_skip(self, reason: str) -> None:
        """Record a skipped row with reason."""
        self.stats['skipped_rows'] += 1
        self.stats['skipped_reasons'][reason] = self.stats['skipped_reasons'].get(reason, 0) + 1
    
    def process_input_file(self, resume_from: int = 0) -> None:
        """
        Process input CSV file row by row.
        
        Args:
            resume_from: Row number to start processing from (0-indexed)
        """
        if not Path(self.input_file).exists():
            print(f"[ERROR] Input file not found: {self.input_file}")
            return
        
        # Initialize output files
        file_handles = self.initialize_output_files()
        writers = {
            category: csv.DictWriter(
                file_handles[category],
                fieldnames=self.OUTPUT_HEADERS
            )
            for category in self.CATEGORIES
        }
        
        last_saved_row = resume_from - 1 if resume_from > 0 else -1
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row_num, row in enumerate(reader):
                    # Skip if already processed
                    if row_num < resume_from:
                        continue
                    
                    # Process row
                    if self.process_row(row, file_handles, writers):
                        self.stats['processed_rows'] += 1
                    
                    # Track absolute row number (0-indexed)
                    self.stats['total_rows'] = row_num + 1
                    last_saved_row = row_num
                    
                    # Periodic progress updates and saves
                    if (row_num + 1) % 1000 == 0:
                        self.save_progress(row_num)
                        print(f"[PROGRESS] Processed {row_num + 1} rows...")
        
        finally:
            # Close all file handles
            for handle in file_handles.values():
                handle.close()
            
            # Save final progress (only if we processed rows in this run)
            if last_saved_row >= resume_from:
                self.save_progress(last_saved_row)
    
    def print_summary(self) -> None:
        """Print processing summary."""
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"Domain: {self.domain}")
        print(f"Input file: {self.input_file}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nStatistics:")
        print(f"  Total rows: {self.stats['total_rows']}")
        print(f"  Processed: {self.stats['processed_rows']}")
        print(f"  Skipped: {self.stats['skipped_rows']}")
        
        if self.stats['skipped_reasons']:
            print(f"\nSkipped reasons:")
            for reason, count in sorted(self.stats['skipped_reasons'].items(), key=lambda x: -x[1]):
                print(f"  - {reason}: {count}")
        
        print(f"\nOutput files:")
        for category in self.CATEGORIES:
            file_path = self.output_files[category]
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"  - {file_path.name} ({size:,} bytes)")
        
        print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process conversational datasets and split by model/turn type"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain name (used for labeled dataset lookup and output directory)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ConversationalPipeline(args.file, args.domain)
    
    print("[PIPELINE] Starting conversational pipeline...")
    print(f"   Input: {args.file}")
    print(f"   Domain: {args.domain}\n")
    
    # Load labeled dataset
    pipeline.load_labeled_dataset()
    
    # Load progress from previous run
    resume_from = pipeline.load_progress()
    if resume_from > 0:
        print(f"[PIPELINE] Resuming from row {resume_from}...\n")
    
    # Process input file
    pipeline.process_input_file(resume_from=resume_from)
    
    # Print summary
    pipeline.print_summary()


if __name__ == "__main__":
    main()
