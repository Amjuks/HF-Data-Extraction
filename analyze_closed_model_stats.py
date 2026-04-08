import os
import pandas as pd

def analyze_closed_model_stats(datasets_dir):
    csv_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the datasets directory.")
        return
    print(f"Found {len(csv_files)} CSV files in '{datasets_dir}'.\n")
    for fname in csv_files:
        path = os.path.join(datasets_dir, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {fname}: {e}")
            continue
        if 'closed_model' not in df.columns:
            print(f"File '{fname}' does not have a 'closed_model' column. Skipping.")
            continue
        total = len(df)
        true_count = df['closed_model'].astype(str).str.lower().eq('true').sum()
        false_count = df['closed_model'].astype(str).str.lower().eq('false').sum()
        none_count = total - true_count - false_count
        true_ratio = true_count / total if total else 0
        false_ratio = false_count / total if total else 0
        print(f"Stats for '{fname}':")
        print(f"  Total rows: {total}")
        print(f"  closed_model = True:  {true_count} ({true_ratio:.2%})")
        print(f"  closed_model = False: {false_count} ({false_ratio:.2%})")
        if none_count > 0:
            print(f"  closed_model = None/Other: {none_count} ({none_count/total:.2%})")
        print()

def main():
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    analyze_closed_model_stats(datasets_dir)

if __name__ == "__main__":
    main()
