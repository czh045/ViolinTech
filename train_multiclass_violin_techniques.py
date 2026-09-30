import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from violin_feature_utils import (
    drop_excluded_training_features,
    drop_metadata_columns,
    extract_audio_features,
    extract_skeleton_feature_set,
    find_matching_csv,
)


CLASS_MAP = {
    "boxian": "pizzicato",
    "chanyin": "tremolo",
    "dungong": "dungong",
    "plain": "plain",
    "rouxian": "vibrato",
    "shuangyin": "double_stop",
    "tiaogong": "spiccato",
}


def prepared_relative_path(wav_path: Path, base_dir: Path, folder_name: str) -> Path | None:
    try:
        return wav_path.relative_to(base_dir / "prepared" / folder_name)
    except ValueError:
        return None


def resolve_source_video_path(wav_path: Path, base_dir: Path, folder_name: str) -> Path:
    relative_path = prepared_relative_path(wav_path, base_dir, folder_name)
    if relative_path is not None:
        return base_dir / folder_name / relative_path.with_suffix(".mp4")
    return wav_path.with_suffix(".mp4")


def source_video_exists_for_prepared_wav(wav_path: Path, base_dir: Path, folder_name: str) -> bool:
    return resolve_source_video_path(wav_path, base_dir, folder_name).exists()


def csv_kind(csv_path: Path) -> str:
    name = csv_path.name.lower()
    if name.endswith("_violin_skeleton.csv"):
        return "violin_skeleton"
    if "mediapipe" in name:
        return "mediapipe"
    if "skeleton" in name:
        return "skeleton"
    return "csv"


def sample_metadata(wav_path: Path, csv_path: Path, base_dir: Path, folder_name: str) -> dict:
    relative_path = prepared_relative_path(wav_path, base_dir, folder_name)
    source_video = resolve_source_video_path(wav_path, base_dir, folder_name)
    if relative_path is None:
        try:
            prepared_relpath = wav_path.relative_to(base_dir).as_posix()
        except ValueError:
            prepared_relpath = wav_path.as_posix()
        sample_key = Path(folder_name) / wav_path.stem
    else:
        prepared_relpath = (Path("prepared") / folder_name / relative_path).as_posix()
        sample_key = Path(folder_name) / relative_path.with_suffix("")

    try:
        source_relpath = source_video.relative_to(base_dir).as_posix()
    except ValueError:
        source_relpath = source_video.as_posix()

    return {
        "sample_id": sample_key.as_posix(),
        "sample_name": wav_path.stem,
        "source_group": source_relpath,
        "source_video_path": source_relpath,
        "source_video_exists": source_video.exists(),
        "prepared_relpath": prepared_relpath,
        "csv_kind": csv_kind(csv_path),
    }


def collect_samples(base_dir: Path, folder_names: list[str]) -> list[dict]:
    samples = []
    skipped_missing_csv = 0
    skipped_bad_features = 0
    skipped_nonuniform_skeleton = 0
    skipped_stale_prepared = 0
    prepared_root = base_dir / "prepared"
    for folder_name in folder_names:
        source_dir = prepared_root / folder_name
        using_prepared = source_dir.exists()
        if not source_dir.exists():
            source_dir = base_dir / folder_name
            using_prepared = False
        if not source_dir.exists():
            print(f"Skip missing folder: {source_dir}")
            continue
        wav_files = sorted(source_dir.rglob("*.wav"))
        if not wav_files:
            print(f"Skip {folder_name}: no wav files")
            continue
        for wav_path in wav_files:
            if using_prepared and not source_video_exists_for_prepared_wav(wav_path, base_dir, folder_name):
                skipped_stale_prepared += 1
                continue
            csv_path = None
            uniform_csv = wav_path.parent / f"{wav_path.stem}_violin_skeleton.csv"
            if using_prepared and not uniform_csv.exists():
                skipped_nonuniform_skeleton += 1
                continue
            for candidate in [
                wav_path.parent / f"{wav_path.stem}_violin_skeleton.csv",
                wav_path.parent / f"{wav_path.stem}_mediapipe.csv",
                wav_path.parent / f"{wav_path.stem}_skeleton.csv",
                wav_path.parent / f"{wav_path.stem}.csv",
            ]:
                if candidate.exists():
                    csv_path = candidate
                    break
            if csv_path is None:
                csv_path = find_matching_csv(wav_path, base_dir=base_dir)
            if csv_path is None:
                skipped_missing_csv += 1
                continue
            audio_feats = extract_audio_features(wav_path)
            skeleton_feats = extract_skeleton_feature_set(csv_path)
            if audio_feats is None or skeleton_feats is None:
                skipped_bad_features += 1
                continue
            samples.append(
                {
                    "label": CLASS_MAP.get(folder_name, folder_name),
                    "folder": folder_name,
                    "wav_path": str(wav_path),
                    "csv_path": str(csv_path),
                    **sample_metadata(wav_path, csv_path, base_dir, folder_name),
                    **audio_feats,
                    **skeleton_feats,
                }
            )
    if skipped_missing_csv:
        print(f"Skipped samples without CSV: {skipped_missing_csv}")
    if skipped_bad_features:
        print(f"Skipped samples with unusable audio/skeleton features: {skipped_bad_features}")
    if skipped_nonuniform_skeleton:
        print(f"Skipped prepared samples without uniform violin skeleton CSV: {skipped_nonuniform_skeleton}")
    if skipped_stale_prepared:
        print(f"Skipped stale prepared samples without source video: {skipped_stale_prepared}")
    return samples


def build_feature_matrix(samples: list[dict]):
    if not samples:
        return None, None, None, None
    df = pd.DataFrame(samples)
    labels = df["label"].tolist()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    feature_frame = drop_metadata_columns(df)
    feature_frame = drop_excluded_training_features(feature_frame)
    X = feature_frame.fillna(0).to_numpy(dtype=float)
    return X, y, feature_frame.columns.tolist(), encoder


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    encoder: LabelEncoder,
    model_path: Path,
) -> None:
    unique, counts = np.unique(y, return_counts=True)
    distribution = {
        encoder.inverse_transform([int(cls)])[0]: int(cnt)
        for cls, cnt in zip(unique, counts)
    }
    print("Class distribution:", distribution)
    if len(unique) < 2:
        raise ValueError("Need at least two classes to train a multiclass model.")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    min_class_count = int(np.min(counts))
    if len(y) >= 12 and min_class_count >= 2:
        cv = StratifiedKFold(n_splits=min(3, min_class_count), shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            scores.append(accuracy_score(y[test_idx], pred))
            print(f"Fold {fold}: accuracy={scores[-1]:.4f}")
        print(f"Mean CV accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
        model.fit(X, y)
    else:
        if min_class_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Train samples:", len(y_train), "test samples:", len(y_test))
        print("Accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=encoder.inverse_transform(np.unique(y_test)), digits=4))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    joblib.dump({"model": model, "feature_names": feature_names, "label_encoder": encoder}, model_path)
    print(f"Saved model to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multiclass violin technique model")
    parser.add_argument("--base-dir", default=r"Z:\ViolinTech", help="Project root")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["boxian", "dungong", "plain", "rouxian", "shuangyin", "tiaogong"],
        help="Technique folders to include",
    )
    parser.add_argument("--model-out", default="violin_technique_model.joblib", help="Model output path")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    samples = collect_samples(base_dir, args.folders)
    print(f"Collected samples: {len(samples)}")
    if not samples:
        raise SystemExit("No usable samples found. Check wav and MediaPipe CSV pairs.")

    X, y, feature_names, encoder = build_feature_matrix(samples)
    print("Feature count:", X.shape[1])
    train_and_evaluate(X, y, feature_names, encoder, Path(args.model_out))


if __name__ == "__main__":
    main()
