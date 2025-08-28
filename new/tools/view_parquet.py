#!/usr/bin/env python3
"""
Simple Parquet viewer.

Loads a Parquet file and prints a compact summary:
- file size, row/column counts
- column names and dtypes
- first N rows

Usage (args also provided via VS Code launch):
  python tools/view_parquet.py --file /path/to/file.parquet --rows 5
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def human_bytes(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def print_header(path: str) -> None:
    print("== Parquet File ==")
    print(f"Path: {path}")
    try:
        size = os.path.getsize(path)
        print(f"Size: {human_bytes(size)}")
    except OSError:
        pass
    print()


def view_with_pandas(path: str, rows: int) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        print(f"[Info] pandas not available ({e!s}). Will try pyarrow fallback.")
        return False

    try:
        df = pd.read_parquet(path)  # requires pyarrow or fastparquet
    except Exception as e:
        print(f"[Warn] pandas failed to read parquet: {e!s}")
        return False

    # Configure display for compact preview
    with pd.option_context(
        'display.max_colwidth', 160,
        'display.max_columns', 80,
        'display.width', 200,
        'display.expand_frame_repr', False
    ):
        print("== Summary (pandas) ==")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns):,}")
        print("Columns:")
        print("  " + ", ".join(map(str, df.columns.tolist())))
        print()
        print("Dtypes:")
        print(df.dtypes)
        print()
        n = min(rows, len(df))
        if n > 0:
            print(f"First {n} rows:")
            print(df.head(n))
        else:
            print("Dataframe is empty.")
    return True


def view_with_pyarrow(path: str, rows: int) -> bool:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        print(f"[Error] Neither pandas nor pyarrow are available to read parquet ({e!s}).")
        print("Please install 'pandas[parquet]' or 'pyarrow'.")
        return False

    try:
        table = pq.read_table(path)
    except Exception as e:
        print(f"[Error] pyarrow failed to read parquet: {e!s}")
        return False

    print("== Summary (pyarrow) ==")
    print(f"Rows: {table.num_rows:,}")
    print(f"Columns: {table.num_columns:,}")
    print("Schema:")
    print(table.schema)
    print()

    n = min(rows, table.num_rows)
    if n > 0:
        try:
            head_tbl = table.slice(0, n)
            # Convert to pandas for prettier print if available
            try:
                import pandas as pd  # noqa: F401
                print(f"First {n} rows:")
                print(head_tbl.to_pandas())
            except Exception:
                print(f"First {n} rows (arrow):")
                print(head_tbl)
        except Exception as e:
            print(f"[Warn] Failed to render head rows: {e!s}")
    else:
        print("Table is empty.")
    return True


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Parquet file viewer")
    parser.add_argument(
        "--file",
        default="/home/dids/shiyang/datasets/OpenThinkIMG-Chart-SFT-2942/data/train-00000-of-00001.parquet",
        help="Path to a .parquet file"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of rows to preview"
    )

    args = parser.parse_args(argv)

    if not os.path.isfile(args.file):
        print(f"[Error] File not found: {args.file}")
        return 2

    print_header(args.file)

    # Try pandas first, fall back to pyarrow summary.
    if view_with_pandas(args.file, args.rows):
        return 0
    if view_with_pyarrow(args.file, args.rows):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
