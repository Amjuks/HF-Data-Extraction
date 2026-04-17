import os
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

TOP_DISPLAY = 10


def make_output_dir(base_dir):
    reports_dir = os.path.join(base_dir, 'model_name_reports')
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir


def normalize_model_name(series):
    return (
        series.fillna('<missing>')
        .astype(str)
        .str.strip()
        .replace({'': '<missing>', 'None': '<missing>', 'nan': '<missing>'})
    )


def save_graph(stats_df, output_path, title):
    if plt is None:
        return False

    top_df = stats_df.head(TOP_DISPLAY)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(top_df))))
    ax.barh(top_df['model_name'], top_df['count'], color='#4c72b0')
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Count')
    ax.set_ylabel('model_name')

    for idx, row in top_df.iterrows():
        ax.text(
            row['count'] + max(1, row['count'] * 0.01),
            idx,
            f"{int(row['count'])} ({row['percentage']:.1f}%)",
            va='center',
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def analyze_model_name_stats(datasets_dir):
    labeled_files = sorted(
        f for f in os.listdir(datasets_dir)
        if f.endswith('_labeled.csv')
    )

    if not labeled_files:
        print("No _labeled.csv files found in the datasets directory.")
        return

    reports_dir = make_output_dir(os.path.dirname(datasets_dir))

    for fname in labeled_files:
        path = os.path.join(datasets_dir, fname)
        domain = fname.replace('_labeled.csv', '')
        print('=' * 80)
        print(f"Domain report: {domain}")
        print('-' * 80)

        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as e:
            print(f"Failed to read '{fname}': {e}")
            print()
            continue

        if 'model_name' not in df.columns:
            print(f"File '{fname}' does not contain a 'model_name' column.")
            print()
            continue

        total_rows = len(df)
        if total_rows == 0:
            print(f"File '{fname}' is empty.")
            print()
            continue

        model_series = normalize_model_name(df['model_name'])
        missing_count = (model_series == '<missing>').sum()
        valid_series = model_series[model_series != '<missing>']

        counts = valid_series.value_counts()
        total_valid = counts.sum()

        if total_valid == 0:
            print(f"No non-missing model_name values found in '{fname}'.")
            print(f"Total rows: {total_rows}")
            print(f"Excluded <missing>: {missing_count}")
            print()
            continue

        percent = (counts / total_valid * 100).round(2)
        stats_df = pd.DataFrame({
            'model_name': counts.index,
            'count': counts.values,
            'percentage': percent.values,
        })

        top_count = stats_df.iloc[0]['count']
        top_percentage = stats_df.iloc[0]['percentage']
        top_models = stats_df.loc[stats_df['count'] == top_count, 'model_name'].tolist()
        top_models_text = ', '.join(top_models)

        print(f"Total rows: {total_rows}")
        print(f"Valid model_name rows: {total_valid}")
        print(f"Excluded <missing>: {missing_count}")
        print(f"Unique model_name values: {len(stats_df)}")
        print(f"Most used model_name: {top_models_text} ({top_count} rows, {top_percentage:.2f}%)")
        print()

        display_df = stats_df.head(TOP_DISPLAY).copy()
        display_df['percentage'] = display_df['percentage'].map('{:.2f}%'.format)
        print(display_df.to_string(index=False, justify='left'))
        print()

        csv_path = os.path.join(reports_dir, f'{domain}_model_name_summary.csv')
        stats_df.to_csv(csv_path, index=False)

        graph_path = os.path.join(reports_dir, f'{domain}_model_name_top{TOP_DISPLAY}.png')
        graph_saved = save_graph(stats_df, graph_path, f"Top {TOP_DISPLAY} model_name counts for {domain}")

        print(f"Saved summary CSV: {csv_path}")
        if graph_saved:
            print(f"Saved graph: {graph_path}")
        else:
            print("matplotlib is not installed; graph file was not generated.")
        print()


def main():
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    analyze_model_name_stats(datasets_dir)


if __name__ == '__main__':
    main()
