from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from train_multiclass_violin_techniques import collect_samples
from violin_feature_utils import METADATA_COLUMNS, drop_excluded_training_features, drop_metadata_columns


AUDIO_PREFIXES = (
    "onset",
    "duration",
    "centroid",
    "bandwidth",
    "zcr",
    "rms",
    "rolloff",
    "tempo",
    "mfcc",
)
SKELETON_PREFIXES = ("r_", "l_", "right_", "left_", "dist_", "bow_")


def detect_boundaries(labels: List[str], min_gap: int = 1) -> List[int]:
    boundaries: List[int] = []
    if not labels:
        return boundaries

    min_gap = max(1, int(min_gap))
    previous = labels[0]
    for idx, label in enumerate(labels[1:], start=1):
        if label == previous:
            continue
        previous = label
        if not boundaries or idx - boundaries[-1] >= min_gap:
            boundaries.append(idx)
    return boundaries


def split_modalities(feature_names: List[str]) -> Tuple[List[str], List[str]]:
    audio_names: List[str] = []
    skeleton_names: List[str] = []
    for name in feature_names:
        if any(name.startswith(prefix) for prefix in SKELETON_PREFIXES):
            skeleton_names.append(name)
        elif any(name.startswith(prefix) for prefix in AUDIO_PREFIXES):
            audio_names.append(name)
        else:
            audio_names.append(name)
    return audio_names, skeleton_names


def build_feature_frame(samples: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    df = pd.DataFrame(samples)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"].tolist())
    feature_frame = drop_metadata_columns(df)
    feature_frame = drop_excluded_training_features(feature_frame)
    feature_frame = feature_frame.fillna(0).astype(float)
    return feature_frame, y, encoder


def build_manifest(samples: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(samples)
    columns = [name for name in METADATA_COLUMNS if name in df.columns]
    manifest = df[columns].copy()
    if "source_video_exists" in manifest.columns:
        manifest["source_video_exists"] = manifest["source_video_exists"].astype(bool)
    return manifest.sort_values(["label", "sample_id"], kind="stable").reset_index(drop=True)


def source_group_audit(samples: List[Dict]) -> Dict:
    df = pd.DataFrame(samples)
    if "source_group" not in df.columns:
        return {"available": False}
    grouped = df.groupby("source_group", dropna=False).agg(
        clips=("label", "size"),
        labels=("label", lambda values: sorted(set(values))),
    )
    duplicated = grouped[grouped["clips"] > 1]
    return {
        "available": True,
        "groups": int(grouped.shape[0]),
        "single_clip_groups": int((grouped["clips"] == 1).sum()),
        "multi_clip_groups": int((grouped["clips"] > 1).sum()),
        "max_clips_per_group": int(grouped["clips"].max()) if not grouped.empty else 0,
        "multi_clip_examples": [
            {"source_group": str(index), "clips": int(row["clips"]), "labels": row["labels"]}
            for index, row in duplicated.head(10).iterrows()
        ],
    }


def make_model(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=random_state,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def usable_stratified_splits(y: np.ndarray, n_splits: int) -> int:
    counts = np.bincount(y)
    usable_splits = min(n_splits, int(counts.min()))
    if usable_splits < 2:
        raise ValueError("Each class needs at least two samples for cross-validation.")
    return usable_splits


def make_splitter(
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    groups: np.ndarray | None = None,
):
    usable_splits = usable_stratified_splits(y, n_splits)
    if groups is None:
        splitter = StratifiedKFold(n_splits=usable_splits, shuffle=True, random_state=random_state)
        return splitter.split, usable_splits, "stratified_clip"

    groups = np.asarray(groups, dtype=str)
    for cls in np.unique(y):
        class_groups = np.unique(groups[y == cls])
        usable_splits = min(usable_splits, len(class_groups))
    if usable_splits < 2:
        raise ValueError("Each class needs at least two source groups for grouped cross-validation.")
    splitter = StratifiedGroupKFold(n_splits=usable_splits, shuffle=True, random_state=random_state)
    return splitter.split, usable_splits, "stratified_source_group"


def summarize_predictions(
    y_true_all: list[int],
    y_pred_all: list[int],
    encoder: LabelEncoder,
) -> Dict:
    labels = np.arange(len(encoder.classes_))
    confusion = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_all,
        y_pred_all,
        labels=labels,
        zero_division=0,
    )
    per_class = {}
    for idx, class_name in enumerate(encoder.classes_):
        per_class[class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_matrix(
    X: np.ndarray,
    y: np.ndarray,
    encoder: LabelEncoder,
    n_splits: int,
    random_state: int,
    groups: np.ndarray | None = None,
) -> Dict:
    split_fn, usable_splits, split_strategy = make_splitter(y, n_splits, random_state, groups)

    fold_rows = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    split_args = (X, y) if groups is None else (X, y, groups)
    for fold, (train_idx, test_idx) in enumerate(split_fn(*split_args), start=1):
        model = make_model(random_state + fold)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        fold_rows.append(
            {
                "fold": fold,
                "train_samples": int(len(train_idx)),
                "test_samples": int(len(test_idx)),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y[test_idx], pred, average="weighted", zero_division=0)),
            }
        )
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(pred.tolist())

    prediction_summary = summarize_predictions(y_true_all, y_pred_all, encoder)
    summary = {
        "split_strategy": split_strategy,
        "n_splits": int(usable_splits),
        "accuracy_mean": float(np.mean([row["accuracy"] for row in fold_rows])),
        "accuracy_std": float(np.std([row["accuracy"] for row in fold_rows])),
        "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
        "balanced_accuracy_std": float(np.std([row["balanced_accuracy"] for row in fold_rows])),
        "macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_rows])),
        "macro_f1_std": float(np.std([row["macro_f1"] for row in fold_rows])),
        "weighted_f1_mean": float(np.mean([row["weighted_f1"] for row in fold_rows])),
        "weighted_f1_std": float(np.std([row["weighted_f1"] for row in fold_rows])),
        "folds": fold_rows,
        **prediction_summary,
    }
    return summary


def evaluate_modalities(
    matrices: Dict[str, pd.DataFrame],
    y: np.ndarray,
    encoder: LabelEncoder,
    n_splits: int,
    random_state: int,
    groups: np.ndarray | None = None,
) -> Dict[str, Dict]:
    evaluated: Dict[str, Dict] = {}
    for name, matrix in matrices.items():
        if matrix.shape[1] == 0:
            continue
        evaluated[name] = evaluate_matrix(
            matrix.to_numpy(dtype=float),
            y,
            encoder,
            n_splits=n_splits,
            random_state=random_state,
            groups=groups,
        )
    return evaluated


def out_of_fold_predictions(
    matrices: Dict[str, pd.DataFrame],
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    groups: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    predictions: Dict[str, np.ndarray] = {}
    for name, matrix in matrices.items():
        if matrix.shape[1] == 0:
            continue
        X = matrix.to_numpy(dtype=float)
        split_fn, _, _ = make_splitter(y, n_splits, random_state, groups)
        split_args = (X, y) if groups is None else (X, y, groups)
        pred_all = np.full(len(y), -1, dtype=int)
        for fold, (train_idx, test_idx) in enumerate(split_fn(*split_args), start=1):
            model = make_model(random_state + fold)
            model.fit(X[train_idx], y[train_idx])
            pred_all[test_idx] = model.predict(X[test_idx])
        predictions[name] = pred_all
    return predictions


def cross_modal_error_analysis(
    sample_df: pd.DataFrame,
    matrices: Dict[str, pd.DataFrame],
    y: np.ndarray,
    encoder: LabelEncoder,
    n_splits: int,
    random_state: int,
    groups: np.ndarray,
) -> Dict:
    predictions = out_of_fold_predictions(matrices, y, n_splits, random_state, groups)
    audio_pred = predictions["audio"]
    skeleton_pred = predictions["skeleton"]
    multimodal_pred = predictions["multimodal"]

    audio_correct = audio_pred == y
    skeleton_correct = skeleton_pred == y
    multimodal_correct = multimodal_pred == y
    disagree = audio_pred != skeleton_pred

    case_rows = []
    for idx in np.where(disagree)[0].tolist():
        case_rows.append(
            {
                "sample_id": sample_df.iloc[idx].get("sample_id", ""),
                "source_group": sample_df.iloc[idx].get("source_group", ""),
                "true_label": encoder.inverse_transform([y[idx]])[0],
                "audio_pred": encoder.inverse_transform([audio_pred[idx]])[0],
                "skeleton_pred": encoder.inverse_transform([skeleton_pred[idx]])[0],
                "multimodal_pred": encoder.inverse_transform([multimodal_pred[idx]])[0],
                "audio_correct": bool(audio_correct[idx]),
                "skeleton_correct": bool(skeleton_correct[idx]),
                "multimodal_correct": bool(multimodal_correct[idx]),
            }
        )

    return {
        "split_strategy": "stratified_source_group",
        "n_samples": int(len(y)),
        "audio_skeleton_disagreement_count": int(np.sum(disagree)),
        "multimodal_correct_when_audio_skeleton_disagree": int(np.sum(disagree & multimodal_correct)),
        "audio_correct_skeleton_wrong": int(np.sum(audio_correct & ~skeleton_correct)),
        "skeleton_correct_audio_wrong": int(np.sum(skeleton_correct & ~audio_correct)),
        "both_single_modalities_correct": int(np.sum(audio_correct & skeleton_correct)),
        "both_single_modalities_wrong": int(np.sum(~audio_correct & ~skeleton_correct)),
        "multimodal_correct_when_audio_correct_skeleton_wrong": int(
            np.sum(audio_correct & ~skeleton_correct & multimodal_correct)
        ),
        "multimodal_correct_when_skeleton_correct_audio_wrong": int(
            np.sum(skeleton_correct & ~audio_correct & multimodal_correct)
        ),
        "disagreement_cases": case_rows,
    }


def run_modality_experiments(
    samples: List[Dict],
    n_splits: int = 3,
    random_state: int = 42,
    group_column: str | None = "source_group",
) -> Tuple[Dict, pd.DataFrame, LabelEncoder, pd.DataFrame]:
    sample_df = pd.DataFrame(samples)
    feature_frame, y, encoder = build_feature_frame(samples)
    audio_names, skeleton_names = split_modalities(feature_frame.columns.tolist())
    matrices = {
        "audio": feature_frame[audio_names],
        "skeleton": feature_frame[skeleton_names],
        "multimodal": feature_frame,
    }

    results = {
        "class_names": encoder.classes_.tolist(),
        "class_distribution": {
            encoder.inverse_transform([idx])[0]: int(count)
            for idx, count in enumerate(np.bincount(y))
        },
        "feature_counts": {
            "audio": len(audio_names),
            "skeleton": len(skeleton_names),
            "multimodal": feature_frame.shape[1],
        },
        "source_group_audit": source_group_audit(samples),
        "modalities": evaluate_modalities(
            matrices,
            y,
            encoder,
            n_splits=n_splits,
            random_state=random_state,
        ),
    }
    if group_column and group_column in sample_df.columns:
        groups = sample_df[group_column].astype(str).to_numpy()
        results["grouped_modalities"] = evaluate_modalities(
            matrices,
            y,
            encoder,
            n_splits=n_splits,
            random_state=random_state,
            groups=groups,
        )
        results["cross_modal_error_analysis"] = cross_modal_error_analysis(
            sample_df,
            matrices,
            y,
            encoder,
            n_splits=n_splits,
            random_state=random_state,
            groups=groups,
        )
    return results, feature_frame, encoder, build_manifest(samples)


def train_final_model(
    feature_frame: pd.DataFrame,
    samples: List[Dict],
    encoder: LabelEncoder,
    model_path: Path,
    random_state: int = 42,
) -> List[Dict]:
    y = encoder.transform(pd.DataFrame(samples)["label"].tolist())
    model = make_model(random_state)
    model.fit(feature_frame.to_numpy(dtype=float), y)
    importances = model.named_steps["classifier"].feature_importances_
    ranked = [
        {"feature": name, "importance": float(score)}
        for name, score in sorted(zip(feature_frame.columns.tolist(), importances), key=lambda item: item[1], reverse=True)
    ]
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_frame.columns.tolist(),
            "label_encoder": encoder,
            "feature_importance": ranked,
        },
        model_path,
    )
    return ranked


def add_result_rows(rows: list[dict], split_name: str, modality_results: Dict[str, Dict]) -> None:
    for modality, summary in modality_results.items():
        rows.append(
            {
                "split": split_name,
                "modality": modality,
                "n_splits": summary["n_splits"],
                "accuracy_mean": summary["accuracy_mean"],
                "accuracy_std": summary["accuracy_std"],
                "balanced_accuracy_mean": summary["balanced_accuracy_mean"],
                "balanced_accuracy_std": summary["balanced_accuracy_std"],
                "macro_f1_mean": summary["macro_f1_mean"],
                "macro_f1_std": summary["macro_f1_std"],
                "weighted_f1_mean": summary["weighted_f1_mean"],
                "weighted_f1_std": summary["weighted_f1_std"],
            }
        )


def paired_statistical_tests(results: Dict) -> list[dict]:
    rows: list[dict] = []
    metrics = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    split_sources = [("stratified_clip", results["modalities"])]
    if "grouped_modalities" in results:
        split_sources.append(("stratified_source_group", results["grouped_modalities"]))

    for split_name, modality_results in split_sources:
        multimodal_folds = modality_results["multimodal"]["folds"]
        for baseline in ("audio", "skeleton"):
            baseline_folds = modality_results[baseline]["folds"]
            for metric in metrics:
                multimodal_scores = np.array([row[metric] for row in multimodal_folds], dtype=float)
                baseline_scores = np.array([row[metric] for row in baseline_folds], dtype=float)
                diff = multimodal_scores - baseline_scores
                if len(diff) > 1:
                    t_stat, p_value = stats.ttest_rel(multimodal_scores, baseline_scores)
                    sem = stats.sem(diff)
                    if np.isfinite(sem) and sem > 0:
                        ci_low, ci_high = stats.t.interval(0.95, len(diff) - 1, loc=float(np.mean(diff)), scale=float(sem))
                    else:
                        ci_low = ci_high = float(np.mean(diff))
                else:
                    t_stat = p_value = ci_low = ci_high = float("nan")
                rows.append(
                    {
                        "split": split_name,
                        "comparison": f"multimodal_vs_{baseline}",
                        "metric": metric,
                        "n_folds": int(len(diff)),
                        "mean_diff": float(np.mean(diff)),
                        "ci95_low": float(ci_low),
                        "ci95_high": float(ci_high),
                        "t_stat": float(t_stat),
                        "p_value": float(p_value),
                    }
                )
    return rows


def write_outputs(results: Dict, importances: List[Dict], manifest: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    statistical_tests = paired_statistical_tests(results)
    results["statistical_tests"] = statistical_tests
    with (out_dir / "paper_experiment_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rows: list[dict] = []
    add_result_rows(rows, "stratified_clip", results["modalities"])
    if "grouped_modalities" in results:
        add_result_rows(rows, "stratified_source_group", results["grouped_modalities"])
    pd.DataFrame(rows).to_csv(out_dir / "paper_experiment_results.csv", index=False)
    pd.DataFrame(statistical_tests).to_csv(out_dir / "paper_statistical_tests.csv", index=False)
    if "cross_modal_error_analysis" in results:
        pd.DataFrame(results["cross_modal_error_analysis"]["disagreement_cases"]).to_csv(
            out_dir / "paper_cross_modal_errors.csv",
            index=False,
        )
    pd.DataFrame(importances).to_csv(out_dir / "paper_feature_importance.csv", index=False)
    manifest.to_csv(out_dir / "data_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-ready multimodal ablation experiments")
    parser.add_argument("--base-dir", default=r"Z:\ViolinTech")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["boxian", "dungong", "plain", "rouxian", "shuangyin", "tiaogong"],
    )
    parser.add_argument("--out-dir", default="tmp_out")
    parser.add_argument("--model-out", default="violin_technique_model.joblib")
    parser.add_argument("--splits", type=int, default=3)
    args = parser.parse_args()

    samples = collect_samples(Path(args.base_dir), args.folders)
    print(f"Usable samples: {len(samples)}")
    if not samples:
        raise SystemExit("No usable samples found.")

    results, feature_frame, encoder, manifest = run_modality_experiments(samples, n_splits=args.splits)
    importances = train_final_model(feature_frame, samples, encoder, Path(args.model_out))
    write_outputs(results, importances, manifest, Path(args.out_dir))

    print("Class distribution:", results["class_distribution"])
    print("Feature counts:", results["feature_counts"])
    print("Source group audit:", results["source_group_audit"])
    for split_name, modality_results in [
        ("stratified_clip", results["modalities"]),
        ("stratified_source_group", results.get("grouped_modalities", {})),
    ]:
        for modality, summary in modality_results.items():
            print(
                f"{split_name}/{modality}: "
                f"accuracy={summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}, "
                f"balanced_accuracy={summary['balanced_accuracy_mean']:.4f} +/- {summary['balanced_accuracy_std']:.4f}, "
                f"macro_f1={summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}"
            )
    print(f"Saved experiment outputs to {Path(args.out_dir)}")
    print(f"Saved final model to {args.model_out}")


if __name__ == "__main__":
    main()
