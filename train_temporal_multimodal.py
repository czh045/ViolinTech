from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from violin_feature_utils import (
    drop_excluded_training_features,
    extract_audio_features_from_array,
    extract_skeleton_feature_set_from_df,
    parse_mediapipe_csv,
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


def source_video_exists_for_prepared_wav(wav_path: Path, base_dir: Path, folder_name: str) -> bool:
    try:
        relative_path = wav_path.relative_to(base_dir / "prepared" / folder_name)
    except ValueError:
        return True
    return (base_dir / folder_name / relative_path.with_suffix(".mp4")).exists()


def find_csv_for_wav(wav_path: Path) -> Path | None:
    for candidate in [
        wav_path.parent / f"{wav_path.stem}_violin_skeleton.csv",
        wav_path.parent / f"{wav_path.stem}_mediapipe.csv",
        wav_path.parent / f"{wav_path.stem}.csv",
    ]:
        if candidate.exists():
            return candidate
    return None


def skeleton_time_column(df: pd.DataFrame) -> str | None:
    for name in ("time", "zeit"):
        if name in df.columns:
            return name
    return None


def slice_skeleton_df(df: pd.DataFrame, start_sec: float, end_sec: float, audio_duration: float) -> pd.DataFrame:
    time_col = skeleton_time_column(df)
    if time_col is not None:
        times = pd.to_numeric(df[time_col], errors="coerce")
        window = df[(times >= start_sec) & (times < end_sec)]
        if not window.empty:
            return window

    if audio_duration <= 0 or df.empty:
        return df.iloc[0:0]
    start_idx = int(np.floor((start_sec / audio_duration) * len(df)))
    end_idx = int(np.ceil((end_sec / audio_duration) * len(df)))
    start_idx = max(0, min(start_idx, len(df)))
    end_idx = max(start_idx + 1, min(end_idx, len(df)))
    return df.iloc[start_idx:end_idx]


def build_temporal_samples(
    base_dir: Path,
    folders: List[str],
    window_sec: float = 1.0,
    hop_sec: float = 0.25,
) -> List[Dict]:
    samples: List[Dict] = []
    skipped_bad_skeleton = 0
    skipped_nonuniform_skeleton = 0
    skipped_stale_prepared = 0
    for folder in folders:
        source_dir = base_dir / "prepared" / folder
        using_prepared = source_dir.exists()
        if not source_dir.exists():
            source_dir = base_dir / folder
            using_prepared = False
        if not source_dir.exists():
            continue
        for wav_path in sorted(source_dir.rglob("*.wav")):
            if using_prepared and not source_video_exists_for_prepared_wav(wav_path, base_dir, folder):
                skipped_stale_prepared += 1
                continue
            if using_prepared and not (wav_path.parent / f"{wav_path.stem}_violin_skeleton.csv").exists():
                skipped_nonuniform_skeleton += 1
                continue
            csv_path = find_csv_for_wav(wav_path)
            if csv_path is None:
                continue
            try:
                y, sr = librosa.load(str(wav_path), sr=22050)
            except Exception:
                continue
            if y.size == 0:
                continue
            df = parse_mediapipe_csv(csv_path)
            if df is None or df.empty:
                skipped_bad_skeleton += 1
                continue

            audio_duration = len(y) / sr
            n_windows = int(np.floor((audio_duration - window_sec) / hop_sec)) + 1
            for idx in range(max(1, n_windows)):
                start_sec = idx * hop_sec
                end_sec = min(audio_duration, start_sec + window_sec)
                start_sample = int(start_sec * sr)
                end_sample = int(end_sec * sr)
                if end_sample - start_sample < int(0.2 * sr):
                    continue

                audio_feats = extract_audio_features_from_array(y[start_sample:end_sample], sr)
                if audio_feats is None:
                    continue
                skeleton_slice = slice_skeleton_df(df, start_sec, end_sec, audio_duration)
                skeleton_feats = extract_skeleton_feature_set_from_df(skeleton_slice)
                if skeleton_feats is None:
                    skipped_bad_skeleton += 1
                    continue
                samples.append(
                    {
                        "label": CLASS_MAP.get(folder, folder),
                        "folder": folder,
                        "wav_path": str(wav_path),
                        "start": start_sec,
                        "end": end_sec,
                        **audio_feats,
                        **skeleton_feats,
                    }
                )
    if skipped_bad_skeleton:
        print(f"Skipped windows with unusable skeleton data: {skipped_bad_skeleton}")
    if skipped_nonuniform_skeleton:
        print(f"Skipped prepared samples without uniform violin skeleton CSV: {skipped_nonuniform_skeleton}")
    if skipped_stale_prepared:
        print(f"Skipped stale prepared samples without source video: {skipped_stale_prepared}")
    return samples


def build_feature_matrix(samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder]:
    df = pd.DataFrame(samples)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"].tolist())
    feature_frame = df.drop(columns=["label", "folder", "wav_path", "start", "end"], errors="ignore")
    feature_frame = drop_excluded_training_features(feature_frame)
    X = feature_frame.fillna(0).to_numpy(dtype=float)
    return X, y, feature_frame.columns.tolist(), encoder


def train_model(X: np.ndarray, y: np.ndarray, feature_names: List[str], encoder: LabelEncoder, model_path: Path) -> None:
    unique, counts = np.unique(y, return_counts=True)
    distribution = {
        encoder.inverse_transform([int(cls)])[0]: int(cnt)
        for cls, cnt in zip(unique, counts)
    }
    print("Class distribution:", distribution)
    if len(unique) < 2:
        raise ValueError("Need at least two classes to train.")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    min_class_count = int(np.min(counts))
    if min_class_count >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            stratify=y,
            random_state=42,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, target_names=encoder.inverse_transform(np.unique(y_test)), digits=4))
    print(confusion_matrix(y_test, pred))

    importances = model.named_steps["classifier"].feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("Top 20 important features:")
    for name, score in ranked[:20]:
        print(f"  {name}: {score:.4f}")

    importance_path = model_path.with_suffix(".feature_importance.json")
    with importance_path.open("w", encoding="utf-8") as f:
        json.dump(
            [{"feature": name, "importance": float(score)} for name, score in ranked],
            f,
            ensure_ascii=False,
            indent=2,
        )

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "label_encoder": encoder,
            "feature_importance": ranked,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")
    print(f"Saved feature importance to {importance_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal multimodal violin technique model")
    parser.add_argument("--base-dir", default=r"Z:\ViolinTech")
    parser.add_argument("--folders", nargs="+", default=["boxian", "dungong", "plain", "rouxian", "shuangyin", "tiaogong"])
    parser.add_argument("--model-out", default="temporal_multimodal_model.joblib")
    args = parser.parse_args()

    samples = build_temporal_samples(Path(args.base_dir), args.folders)
    print(f"Temporal samples: {len(samples)}")
    if not samples:
        raise SystemExit("No temporal samples were generated.")
    X, y, feature_names, encoder = build_feature_matrix(samples)
    train_model(X, y, feature_names, encoder, Path(args.model_out))


if __name__ == "__main__":
    main()
