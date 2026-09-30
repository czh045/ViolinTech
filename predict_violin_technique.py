import argparse
from pathlib import Path

import joblib
import numpy as np

from violin_feature_utils import extract_audio_features, extract_skeleton_feature_set, find_matching_csv


LABEL_NAMES = {
    "pizzicato": "拨弦",
    "dungong": "顿弓",
    "plain": "无技巧",
    "vibrato": "揉弦",
    "double_stop": "双音",
    "spiccato": "跳弓",
    "tremolo": "颤音",
}


def find_preferred_csv(wav_path: Path) -> Path | None:
    for candidate in [
        wav_path.parent / f"{wav_path.stem}_violin_skeleton.csv",
        wav_path.parent / f"{wav_path.stem}_mediapipe.csv",
    ]:
        if candidate.exists():
            return candidate
    return find_matching_csv(wav_path)


def build_sample(wav_path: Path, csv_path: Path | None = None) -> dict:
    audio_feats = extract_audio_features(wav_path)
    if audio_feats is None:
        raise ValueError(f"Cannot read audio: {wav_path}")
    if csv_path is None:
        csv_path = find_preferred_csv(wav_path)
    if csv_path is None or not csv_path.exists():
        raise ValueError(f"Cannot find matching skeleton CSV for: {wav_path}")
    skeleton_feats = extract_skeleton_feature_set(csv_path)
    if skeleton_feats is None:
        raise ValueError(f"Cannot read usable skeleton CSV: {csv_path}")
    return {**audio_feats, **skeleton_feats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict violin technique class")
    parser.add_argument("--wav", required=True, help="Input wav file")
    parser.add_argument("--csv", default=None, help="Matching skeleton CSV file")
    parser.add_argument("--model", default="violin_technique_model.joblib", help="Model file")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model file does not exist: {model_path}")

    data = build_sample(Path(args.wav), Path(args.csv) if args.csv else None)
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    encoder = model_bundle["label_encoder"]
    sample = np.array([[data.get(name, 0.0) for name in feature_names]], dtype=float)

    pred_idx = model.predict(sample)[0]
    label = encoder.inverse_transform([pred_idx])[0]
    display = LABEL_NAMES.get(label, label)
    print(f"Predicted technique: {label} ({display})")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)[0]
        print("Class probabilities:")
        for idx, prob in sorted(zip(np.arange(len(probs)), probs), key=lambda kv: kv[1], reverse=True):
            class_name = encoder.inverse_transform([idx])[0]
            print(f"  {class_name} ({LABEL_NAMES.get(class_name, class_name)}): {prob:.4f}")


if __name__ == "__main__":
    main()
