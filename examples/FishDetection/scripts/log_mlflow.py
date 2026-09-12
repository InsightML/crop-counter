"""Push one finished run dir to the hosted MLflow — and keep MLflow out of the library.

Deliberately a script, not a ``cropcounter`` module: the package must stay
runnable with no tracking server, no credentials and no mlflow install. Auth is
the env-var contract from the InsightML MLflow guide
(``MLFLOW_TRACKING_URI`` / ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``).

Re-runnable: it starts a fresh run named after the directory each time.

Usage::

    python log_mlflow.py --run-dir runs/cfd17_unfrozen_s0 \\
        --results-summary results/results_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="log_mlflow.py", description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-summary", type=Path, default=None)
    parser.add_argument("--experiment", default="crop-counter_CFD17")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    return parser


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.tracking_uri:
        raise SystemExit("no tracking URI: pass --tracking-uri or set MLFLOW_TRACKING_URI")

    import mlflow

    run_dir = args.run_dir
    name = run_dir.name
    config = read_json(run_dir / "config.json") or {}
    history = read_json(run_dir / "history.json") or {}

    metrics = {f"last_{key}": float(values[-1])
               for key, values in history.items()
               if values and isinstance(values[-1], (int, float)) and values[-1] == values[-1]}

    best = run_dir / "best.pt"
    if best.exists():
        import torch

        payload = torch.load(best, map_location="cpu", weights_only=False)
        for key in ("best_epoch", "best_metric", "epoch"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        del payload

    summary = read_json(args.results_summary) if args.results_summary else None
    for suffix in ("best", "last"):
        scope = (((summary or {}).get("models", {}).get(f"{name}_{suffix}", {})
                  .get("scopes", {}).get("full")) or {})
        for key in ("ap", "ap50", "ap75", "ar100", "best_tau_mae", "mae_at_best", "f1_at_best_tau"):
            if key in scope:
                metrics[f"eval_{suffix}_{key}"] = float(scope[key])

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: v for k, v in config.items() if not isinstance(v, (dict, list))})
        mlflow.log_metrics(metrics)
        artifacts = [run_dir / "history.json", run_dir / "curves.png", run_dir / "config.json"]
        if args.results_summary:
            artifacts += [args.results_summary, args.results_summary.parent / "results.csv"]
        for path in artifacts:  # .pt files are deliberately NOT logged
            if path.exists():
                mlflow.log_artifact(str(path))
    print(f"logged {name} to {args.tracking_uri} / {args.experiment} ({len(metrics)} metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
