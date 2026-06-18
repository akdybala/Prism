"""Train a LightGBM router on generated trusted-v1 signal features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


LABELS = ["light", "medium", "heavy"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def matrix(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(row.get(name, 0) or 0) for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([int(row["route_label_id"]) for row in rows], dtype=np.int64)
    weights = np.asarray(
        [float(row.get("route_label_sample_weight", 1.0) or 1.0) for row in rows],
        dtype=np.float32,
    )
    return x, y, weights


def metrics(model: LGBMClassifier, rows: list[dict], feature_names: list[str]) -> dict:
    x, y, weights = matrix(rows, feature_names)
    pred = model.predict(x)
    proba = model.predict_proba(x)
    report = classification_report(
        y,
        pred,
        labels=[0, 1, 2],
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "rows": len(rows),
        "label_counts": Counter(row["route_label"] for row in rows),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted")), 4),
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        },
        "per_label": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "mean_predicted_probability_for_gold": round(
            float(np.mean(proba[np.arange(len(y)), y])),
            4,
        ),
        "mean_sample_weight": round(float(np.mean(weights)), 4),
    }


def top_importances(model: LGBMClassifier, feature_names: list[str], limit: int = 50) -> list[dict]:
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    rows = []
    for name, gain_value, split_value in zip(feature_names, gain, split):
        rows.append({
            "feature": name,
            "gain": round(float(gain_value), 6),
            "split": int(split_value),
        })
    return sorted(rows, key=lambda item: item["gain"], reverse=True)[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("router_training_data/trusted_v1/router_training_trusted_v1.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("router_training_data/trusted_v1/lightgbm_router"),
    )
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--min-child-samples", type=int, default=35)
    parser.add_argument("--feature-fraction", type=float, default=0.85)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    feature_names = [name for name in rows[0] if name.startswith("feature.")]
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    x_train, y_train, w_train = matrix(train_rows, feature_names)
    x_val, y_val, w_val = matrix(validation_rows, feature_names)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(LABELS),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        min_child_samples=args.min_child_samples,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(x_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="multi_logloss",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "lightgbm_router.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "labels": LABELS,
            "input": str(args.input),
        },
        model_path,
    )
    report = {
        "input": str(args.input),
        "model_path": str(model_path),
        "feature_count": len(feature_names),
        "params": model.get_params(),
        "train": metrics(model, train_rows, feature_names),
        "validation": metrics(model, validation_rows, feature_names),
        "test": metrics(model, test_rows, feature_names),
        "top_feature_importances": top_importances(model, feature_names),
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
