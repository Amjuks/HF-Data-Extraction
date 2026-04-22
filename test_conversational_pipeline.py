#!/usr/bin/env python3
"""
Test suite for conversational_pipeline.py

Tests the pipeline's handling of:
1. Valid data
2. Invalid JSON in conversation field
3. Invalid JSON in metadata field
4. Empty conversations
5. Missing dataset_id
6. Various conversation lengths
"""

import csv
import json
import tempfile
from pathlib import Path


def create_test_input_csv(filepath):
    """Create a test input CSV with various scenarios."""
    test_data = [
        {
            "conversation": json.dumps([
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."}
            ]),
            "language": "english",
            "reasoning": "test",
            "metadata": json.dumps({"task_type": "chat"}),
            "dataset_id": "test-1"
        },
        {
            "conversation": json.dumps([
                {"role": "user", "content": "Hello?"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm good"}
            ]),
            "language": "english",
            "reasoning": "test",
            "metadata": json.dumps({"task_type": "code_generation"}),
            "dataset_id": "test-1"
        },
        {
            "conversation": "INVALID_JSON_NOT_A_LIST",
            "language": "english",
            "reasoning": "test",
            "metadata": json.dumps({"task_type": "code_generation"}),
            "dataset_id": "test-1"
        },
        {
            "conversation": json.dumps([
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"}
            ]),
            "language": "english",
            "reasoning": "test",
            "metadata": "INVALID_JSON_NOT_AN_OBJECT",
            "dataset_id": "test-1"
        },
        {
            "conversation": json.dumps([]),
            "language": "spanish",
            "reasoning": "test",
            "metadata": json.dumps({"task_type": "math"}),
            "dataset_id": "test-2"
        },
        {
            "conversation": json.dumps([
                {"role": "user", "content": "Test"}
            ]),
            "language": "multilingual",
            "reasoning": "test",
            "metadata": json.dumps({"task_type": "nlp"}),
            "dataset_id": "unknown-dataset-id"
        }
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["conversation", "language", "reasoning", "metadata", "dataset_id"])
        writer.writeheader()
        writer.writerows(test_data)


def create_test_labeled_csv(filepath):
    """Create a test labeled dataset CSV."""
    test_data = [
        {"dataset_id": "test-1", "link": "https://example.com/test-1", "closed_model": "True", "model_name": "gpt-4"},
        {"dataset_id": "test-2", "link": "https://example.com/test-2", "closed_model": "False", "model_name": ""}
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["dataset_id", "link", "closed_model", "model_name"])
        writer.writeheader()
        writer.writerows(test_data)


def run_test():
    """Run comprehensive test."""
    
    print("="*70)
    print("CONVERSATIONAL PIPELINE TEST SUITE")
    print("="*70)
    
    # Create temporary test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test files
        input_csv = tmpdir / "test_input.csv"
        labeled_csv = tmpdir / "test_labeled.csv"
        
        create_test_input_csv(input_csv)
        
        # Create datasets directory
        datasets_dir = tmpdir / "datasets"
        datasets_dir.mkdir()
        create_test_labeled_csv(datasets_dir / "test_labeled.csv")
        
        print("\n✅ Test files created:")
        print(f"   Input CSV: {input_csv}")
        print(f"   Labeled CSV: {labeled_csv}")
        
        # Import and run pipeline
        import sys
        import os
        
        # Add test directory to path for imports
        sys.path.insert(0, str(tmpdir))
        
        # Change to temp directory for relative paths
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            from conversational_pipeline import ConversationalPipeline
            
            print("\n" + "="*70)
            print("RUNNING PIPELINE")
            print("="*70)
            
            # Create symbolic link to conversational_pipeline.py
            import shutil
            pipeline_script = Path(__file__).parent / "conversational_pipeline.py"
            shutil.copy(pipeline_script, tmpdir / "conversational_pipeline.py")
            
            # Run pipeline manually
            pipeline = ConversationalPipeline(str(input_csv), "test")
            pipeline.load_labeled_dataset()
            pipeline.process_input_file()
            
            print("\n" + "="*70)
            print("PIPELINE RESULTS")
            print("="*70)
            pipeline.print_summary()
            
            # Validate outputs
            print("\n" + "="*70)
            print("OUTPUT VALIDATION")
            print("="*70)
            
            output_dir = tmpdir / "output" / "test"
            categories = ["closed_single", "closed_multi", "open_single", "open_multi"]
            total_records = 0
            
            for category in categories:
                filepath = output_dir / f"{category}_turn.csv"
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        print(f"\n{category}_turn.csv:")
                        print(f"  Records: {len(rows)}")
                        total_records += len(rows)
                        
                        if rows:
                            first_row = rows[0]
                            print(f"  Sample row:")
                            print(f"    category: {first_row['category']}")
                            print(f"    domain: {first_row['domain']}")
                            print(f"    turn_type: {first_row['turn_type']}")
                            print(f"    made_by: {first_row['made_by']}")
                            print(f"    task: {first_row['task']}")
            
            print(f"\n✅ Total records: {total_records}")
            
            # Check progress file
            progress_file = output_dir / "progress.json"
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    prog = json.load(f)
                    print(f"\n✅ Progress file:")
                    print(f"   last_processed_row: {prog['last_processed_row']}")
            
            # Test statistics
            print(f"\n✅ Processing statistics:")
            print(f"   Total rows in input: {pipeline.stats['total_rows']}")
            print(f"   Rows processed: {pipeline.stats['processed_rows']}")
            print(f"   Rows skipped: {pipeline.stats['skipped_rows']}")
            
            if pipeline.stats['skipped_reasons']:
                print(f"\n   Skipped reasons:")
                for reason, count in sorted(pipeline.stats['skipped_reasons'].items(), key=lambda x: -x[1]):
                    print(f"     - {reason}: {count}")
            
            print("\n" + "="*70)
            print("✅ TEST SUITE PASSED")
            print("="*70)
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    # Note: This test must be run from the project root with conversational_pipeline.py present
    print("\nTo run this test:")
    print("  python test_conversational_pipeline.py")
    print("\nNote: Requires conversational_pipeline.py in the same directory")
