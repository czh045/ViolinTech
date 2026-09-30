import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path

from violin_feature_utils import extract_audio_features, extract_skeleton_feature_set


CLASS_MAP = {
    "boxian": "pizzicato",
    "chanyin": "tremolo",
    "dungong": "dungong",
    "plain": "plain",
    "rouxian": "vibrato",
    "shuangyin": "double_stop",
    "tiaogong": "spiccato",
}


def parse_float(value: str) -> float | None:
    value = str(value).strip().replace(",", ".")
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def csv_has_usable_landmarks(csv_path: Path) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    try:
        handle = csv_path.open("r", newline="", encoding="utf-8-sig")
    except UnicodeDecodeError:
        handle = csv_path.open("r", newline="", encoding="gbk", errors="ignore")
    with handle:
        reader = csv.reader(handle, delimiter=";")
        try:
            header = next(reader)
        except StopIteration:
            return False
        landmark_indexes = [
            idx for idx, name in enumerate(header)
            if name.endswith(("_x3D", "_y3D", "_z3D"))
        ]
        if not landmark_indexes:
            return False
        for row in reader:
            for idx in landmark_indexes:
                if idx >= len(row):
                    continue
                value = parse_float(row[idx])
                if value is not None and abs(value) > 1e-12:
                    return True
    return False


def same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def skeleton_candidate_paths(video_path: Path, output_dir: Path) -> list[Path]:
    stem = video_path.stem
    names = [
        f"{stem}_mediapipe.csv",
        f"{stem}_mediapipe .csv",
        f"{stem}_violin_skeleton.csv",
        f"{stem}_skeleton.csv",
        f"{stem}.csv",
    ]
    candidates: list[Path] = []
    seen: set[str] = set()
    for directory in (output_dir, video_path.parent):
        for name in names:
            candidate = directory / name
            key = str(candidate).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
        if directory.exists():
            for candidate in sorted(directory.glob(f"{stem}*.csv")):
                lower = candidate.name.lower()
                if (
                    "mediapipe" in lower
                    or "skeleton" in lower
                    or candidate.stem == stem
                ):
                    key = str(candidate).lower()
                    if key not in seen:
                        seen.add(key)
                        candidates.append(candidate)
    return candidates


def find_or_copy_existing_csv(video_path: Path, output_dir: Path) -> Path | None:
    target = output_dir / f"{video_path.stem}_mediapipe.csv"
    if csv_has_usable_landmarks(target):
        print(f"Reuse skeleton CSV: {target}", flush=True)
        return target
    for candidate in skeleton_candidate_paths(video_path, output_dir):
        if not csv_has_usable_landmarks(candidate):
            continue
        if not same_path(candidate, target):
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            print(f"Copied skeleton CSV: {candidate} -> {target}", flush=True)
        else:
            print(f"Reuse skeleton CSV: {target}", flush=True)
        return target
    return None


def extract_skeleton_csv(video_path: Path, output_dir: Path) -> Path | None:
    output_csv = output_dir / f"{video_path.stem}_violin_skeleton.csv"
    target = output_dir / f"{video_path.stem}_mediapipe.csv"
    print(f"Extract skeleton: {video_path}", flush=True)
    try:
        from extract_violin_skeleton import process_video

        process_video(video_path, output_csv)
    except Exception as exc:
        print(f"Skeleton extraction failed for {video_path}: {exc}", flush=True)
        return None
    if csv_has_usable_landmarks(output_csv):
        shutil.copy2(output_csv, target)
        print(f"Copied skeleton CSV: {output_csv} -> {target}", flush=True)
        return target
    return find_or_copy_existing_csv(video_path, output_dir)


def ensure_audio_and_csv(
    video_path: Path,
    output_dir: Path,
    force_audio: bool = False,
    force_skeleton: bool = False,
) -> tuple[Path | None, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{video_path.stem}.wav"
    print(f"Prepare sample: {video_path}", flush=True)
    if force_audio or not wav_path.exists():
        print(f"Extract audio: {video_path}", flush=True)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "22050",
            str(wav_path),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            print(f"Audio extraction failed for {video_path}: {exc}", flush=True)
            return None, None

    csv_path = None if force_skeleton else find_or_copy_existing_csv(video_path, output_dir)
    if csv_path is None:
        csv_path = extract_skeleton_csv(video_path, output_dir)
    return wav_path if wav_path.exists() else None, csv_path


def collect_prepared_samples(
    base_dir: Path,
    folder_names: list[str],
    work_dir: Path,
    validate_features: bool = False,
    force_audio: bool = False,
    force_skeleton: bool = False,
) -> list[dict]:
    samples = []
    for folder_name in folder_names:
        source_dir = base_dir / folder_name
        if not source_dir.exists():
            print(f"Skip missing folder: {source_dir}", flush=True)
            continue
        video_paths = sorted(source_dir.rglob("*.mp4"))
        print(f"Folder {folder_name}: {len(video_paths)} videos", flush=True)
        for index, video_path in enumerate(video_paths, start=1):
            print(f"[{folder_name} {index}/{len(video_paths)}]", flush=True)
            target_dir = work_dir / folder_name / video_path.parent.relative_to(source_dir)
            wav_path, csv_path = ensure_audio_and_csv(
                video_path,
                target_dir,
                force_audio=force_audio,
                force_skeleton=force_skeleton,
            )
            if wav_path is None or csv_path is None:
                continue
            if not validate_features:
                samples.append(
                    {
                        "label": CLASS_MAP.get(folder_name, folder_name),
                        "folder": folder_name,
                        "wav_path": str(wav_path),
                        "csv_path": str(csv_path),
                    }
                )
                continue
            audio_feats = extract_audio_features(wav_path)
            skeleton_feats = extract_skeleton_feature_set(csv_path)
            if audio_feats is None or skeleton_feats is None:
                print(f"Skip unusable features: wav={wav_path}, csv={csv_path}", flush=True)
                continue
            samples.append(
                {
                    "label": CLASS_MAP.get(folder_name, folder_name),
                    "folder": folder_name,
                    "wav_path": str(wav_path),
                    "csv_path": str(csv_path),
                    **audio_feats,
                    **skeleton_feats,
                }
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare violin multimodal samples")
    parser.add_argument("--base-dir", default=r"Z:\ViolinTech", help="Project root")
    parser.add_argument("--work-dir", default=r"Z:\ViolinTech\prepared", help="Prepared output directory")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["boxian", "dungong", "plain", "rouxian", "shuangyin", "tiaogong"],
    )
    parser.add_argument(
        "--validate-features",
        action="store_true",
        help="Also compute audio and skeleton features after preparing files",
    )
    parser.add_argument(
        "--force-skeleton",
        action="store_true",
        help="Re-extract skeleton CSV even when a usable CSV already exists",
    )
    parser.add_argument(
        "--force-audio",
        action="store_true",
        help="Re-extract wav audio even when the target wav already exists",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_prepared_samples(
        base_dir,
        args.folders,
        work_dir,
        validate_features=args.validate_features,
        force_audio=args.force_audio,
        force_skeleton=args.force_skeleton,
    )
    print(f"Prepared samples: {len(samples)}")


if __name__ == "__main__":
    main()
