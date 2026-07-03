"""Inspect label distributions for old or new ManekiNeko targets.

Examples:
    python -m manekineko.scripts.inspect_labels \
        --csv xGboost/all-coin.csv \
        --label signal \
        --by tag

    python -m manekineko.scripts.inspect_labels \
        --csv data/features.csv \
        --apply-triple-barrier \
        --horizon 48 \
        --profit-barrier 0.012 \
        --loss-barrier -0.006
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from manekineko.labeling import TripleBarrierConfig, apply_triple_barrier


def _print_distribution(df: pd.DataFrame, label_col: str, by: str | None = None) -> None:
    if label_col not in df.columns:
        raise KeyError(f"label column not found: {label_col}")

    counts = df[label_col].value_counts(dropna=False).sort_index()
    ratio = df[label_col].value_counts(dropna=False, normalize=True).sort_index()

    summary = pd.DataFrame({"count": counts, "ratio": ratio})
    print("\nOverall label distribution")
    print(summary.to_string())

    if by:
        if by not in df.columns:
            raise KeyError(f"group column not found: {by}")
        grouped = (
            df.groupby(by, dropna=False)[label_col]
            .value_counts(dropna=False, normalize=True)
            .rename("ratio")
            .reset_index()
            .sort_values([by, label_col])
        )
        print(f"\nLabel ratio grouped by {by}")
        print(grouped.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Input feature CSV path")
    parser.add_argument("--label", default="signal", help="Existing label column to inspect")
    parser.add_argument("--by", default=None, help="Optional grouping column, e.g. tag or symbol")

    parser.add_argument(
        "--apply-triple-barrier",
        action="store_true",
        help="Generate triple-barrier labels before inspection",
    )
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--high-col", default="high")
    parser.add_argument("--low-col", default="low")
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--profit-barrier", type=float, default=0.012)
    parser.add_argument("--loss-barrier", type=float, default=-0.006)
    parser.add_argument("--fee-round-trip", type=float, default=0.002)
    parser.add_argument("--slippage-round-trip", type=float, default=0.001)
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path to save labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    label_col = args.label
    if args.apply_triple_barrier:
        cfg = TripleBarrierConfig(
            horizon=args.horizon,
            profit_barrier=args.profit_barrier,
            loss_barrier=args.loss_barrier,
            fee_rate_round_trip=args.fee_round_trip,
            slippage_round_trip=args.slippage_round_trip,
        )
        df = apply_triple_barrier(
            df,
            close_col=args.close_col,
            high_col=args.high_col,
            low_col=args.low_col,
            config=cfg,
            prefix="tb",
        )
        label_col = "tb_label"

    _print_distribution(df, label_col=label_col, by=args.by)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved output to {args.output_csv}")


if __name__ == "__main__":
    main()
