from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate a CSV or Parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input file path (.csv or .parquet)")
    parser.add_argument(
        "--output",
        default="",
        help="Output file path (.csv or .parquet). If omitted, '<input>_dedup.<ext>' is used.",
    )
    parser.add_argument(
        "--subset",
        default="",
        help="Comma-separated column names used as dedup keys. If omitted, full-row dedup is used.",
    )
    parser.add_argument(
        "--keep",
        choices=["first", "last", "none"],
        default="first",
        help="Which duplicate to keep. 'none' removes all duplicated groups.",
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Case-insensitive matching for subset string columns.",
    )
    parser.add_argument(
        "--strip-whitespace",
        action="store_true",
        help="Trim leading/trailing whitespace for subset string columns.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file. If set, --output is ignored.",
    )
    return parser.parse_args()


def infer_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext == ".parquet":
        return "parquet"
    raise ValueError(f"Unsupported file extension: {path.suffix}. Use .csv or .parquet")


def load_file(path: Path) -> pd.DataFrame:
    fmt = infer_format(path)
    if fmt == "csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def save_file(df: pd.DataFrame, path: Path) -> None:
    fmt = infer_format(path)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_dedup{input_path.suffix}")


def sanitize_subset_values(df: pd.DataFrame, subset: list[str], ignore_case: bool, strip_whitespace: bool) -> pd.DataFrame:
    if not subset:
        return df
    working = df.copy()
    for col in subset:
        if col not in working.columns:
            raise ValueError(f"Column '{col}' not found in input file.")
        if working[col].dtype == object:
            series = working[col]
            if strip_whitespace:
                series = series.map(lambda x: x.strip() if isinstance(x, str) else x)
            if ignore_case:
                series = series.map(lambda x: x.lower() if isinstance(x, str) else x)
            working[col] = series
    return working


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = input_path if args.inplace else Path(args.output).resolve() if args.output else default_output_path(input_path)
    subset = [x.strip() for x in args.subset.split(",") if x.strip()]
    keep_value: str | bool = False if args.keep == "none" else args.keep

    print(f"Reading: {input_path}")
    df = load_file(input_path)
    before = len(df)
    print(f"Rows before dedup: {before}")

    working_df = sanitize_subset_values(
        df=df,
        subset=subset,
        ignore_case=args.ignore_case,
        strip_whitespace=args.strip_whitespace,
    )
    deduped_index = working_df.drop_duplicates(subset=subset or None, keep=keep_value).index
    deduped_df = df.loc[deduped_index].copy()

    after = len(deduped_df)
    removed = before - after
    print(f"Rows after dedup:  {after}")
    print(f"Rows removed:      {removed}")

    print(f"Writing: {output_path}")
    save_file(deduped_df, output_path)
    print("Done.")


if __name__ == "__main__":
    main()

