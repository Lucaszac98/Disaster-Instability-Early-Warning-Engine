from __future__ import annotations

import argparse
from pathlib import Path

from . import config, data_prep
from .train_model import train_risk_model
from .evaluate import evaluate


def cmd_prepare_data(args: argparse.Namespace) -> None:
    df = data_prep.load_processed()
    print(f"Prepared processed dataset with {len(df)} rows.")
    print("Saved cache to data/processed/disaster_instability.parquet")
    print("Example columns:")
    print(", ".join(list(df.columns)[:20]))


def cmd_train(args: argparse.Namespace) -> None:
    res = train_risk_model(random_state=args.seed)
    print(f"Saved model to: {res.model_path}")
    print("Metrics:")
    print(res.metrics)


def cmd_evaluate(args: argparse.Namespace) -> None:
    out_path = config.REPORTS_METRICS_DIR / "risk_model_metrics.json"
    m = evaluate(out_path=out_path, random_state=args.seed)
    print("Metrics:")
    print(m)
    print(f"Saved to: {out_path}")


def cmd_export_snapshot(args: argparse.Namespace) -> None:
    df = data_prep.load_processed()
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved snapshot to: {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Disaster Instability Early Warning Engine CLI")
    sp = p.add_subparsers(dest="command", required=True)

    p1 = sp.add_parser("prepare-data", help="Compute forces + instability + zones and cache parquet.")
    p1.set_defaults(func=cmd_prepare_data)

    p2 = sp.add_parser("train", help="Train ML risk model for major-disaster escalation.")
    p2.add_argument("--seed", type=int, default=42)
    p2.set_defaults(func=cmd_train)

    p3 = sp.add_parser("evaluate", help="Evaluate trained risk model and save metrics JSON.")
    p3.add_argument("--seed", type=int, default=42)
    p3.set_defaults(func=cmd_evaluate)

    p4 = sp.add_parser("export-snapshot", help="Export processed dataset snapshot to CSV.")
    p4.add_argument("--out", type=Path, default=config.REPORTS_METRICS_DIR / "disaster_instability_snapshot.csv")
    p4.set_defaults(func=cmd_export_snapshot)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
