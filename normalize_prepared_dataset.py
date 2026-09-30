import csv
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREPARED = ROOT / "prepared"
FOLDERS = ["boxian", "dungong", "plain", "rouxian", "shuangyin", "tiaogong"]


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


def candidate_csvs(wav_path: Path, folder: str) -> list[Path]:
    stem = wav_path.stem
    names = [
        f"{stem}_mediapipe.csv",
        f"{stem}_mediapipe .csv",
        f"{stem}_violin_skeleton.csv",
        f"{stem}_skeleton.csv",
        f"{stem}.csv",
    ]
    directories = [wav_path.parent]
    try:
        relative_parent = wav_path.parent.relative_to(PREPARED / folder)
    except ValueError:
        relative_parent = None
    if relative_parent is not None:
        directories.append(ROOT / folder / relative_parent)

    candidates: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
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


def normalize() -> tuple[int, list[Path]]:
    updated = 0
    still_missing: list[Path] = []
    for folder in FOLDERS:
        folder_path = PREPARED / folder
        if not folder_path.exists():
            continue
        for wav_path in sorted(folder_path.rglob("*.wav")):
            target = wav_path.parent / f"{wav_path.stem}_mediapipe.csv"
            if csv_has_usable_landmarks(target):
                continue
            replacement = None
            for candidate in candidate_csvs(wav_path, folder):
                if csv_has_usable_landmarks(candidate):
                    replacement = candidate
                    break
            if replacement is None:
                still_missing.append(target)
                continue
            if not same_path(replacement, target):
                shutil.copy2(replacement, target)
                updated += 1
    return updated, still_missing


if __name__ == "__main__":
    updated_count, missing_paths = normalize()
    print(f"normalize_prepared_dataset completed: updated={updated_count}, still_missing={len(missing_paths)}")
    for path in missing_paths[:20]:
        print(f"missing usable skeleton: {path}")
    if len(missing_paths) > 20:
        print(f"... {len(missing_paths) - 20} more")
