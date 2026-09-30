import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from scipy.fft import dct


METADATA_COLUMNS = (
    "label",
    "folder",
    "sample_id",
    "sample_name",
    "source_group",
    "source_video_path",
    "source_video_exists",
    "prepared_relpath",
    "wav_path",
    "csv_path",
    "csv_kind",
)


def drop_metadata_columns(feature_frame: pd.DataFrame) -> pd.DataFrame:
    metadata = [name for name in feature_frame.columns if name in METADATA_COLUMNS]
    return feature_frame.drop(columns=metadata, errors="ignore")


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-z]+", "", name.lower())
    return normalized


def parse_mediapipe_csv(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    try:
        raw = csv_path.read_text(encoding="utf-8").replace("\ufeff", "")
    except UnicodeDecodeError:
        raw = csv_path.read_text(encoding="gbk", errors="ignore").replace("\ufeff", "")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None
    header = [h.strip() for h in lines[0].split(";")]
    rows = []
    for line in lines[1:]:
        cells = [cell.strip().replace(",", ".") for cell in line.split(";")]
        if len(cells) != len(header):
            continue
        row = []
        for value in cells:
            try:
                row.append(float(value))
            except ValueError:
                row.append(np.nan)
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=header)


def safe_array(series: pd.Series) -> np.ndarray:
    if series is None:
        return np.array([])
    return series.dropna().astype(float).to_numpy()


def is_excluded_training_feature(name: str) -> bool:
    return name == "duration" or name.endswith("_count") or name.endswith("_path_length")


def drop_excluded_training_features(feature_frame: pd.DataFrame) -> pd.DataFrame:
    excluded = [name for name in feature_frame.columns if is_excluded_training_feature(name)]
    return feature_frame.drop(columns=excluded, errors="ignore")


def has_usable_skeleton_data(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    landmark_cols = [
        col for col in df.columns
        if col.endswith(("_x3D", "_y3D", "_z3D"))
    ]
    if not landmark_cols:
        return False
    numeric = df[landmark_cols].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return False
    return bool(np.nanmax(np.abs(values)) > 1e-12)


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def get_coords(df: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alias_map = {
        "r_wrist": ["r_wrist_x3D"],
        "r_elbow": ["r_elbow_x3D"],
        "r_shoulder": ["r_shoulder_x3D"],
        "l_wrist": ["l_wrist_x3D"],
        "l_elbow": ["l_elbow_x3D"],
        "l_shoulder": ["l_shoulder_x3D"],
        "right_index_tip": ["r_indexfinger_tip_x3D", "r_indexfinger1_x3D"],
        "right_middle_tip": ["r_middlefinger_tip_x3D", "r_middlefinger1_x3D"],
        "right_ring_tip": ["r_ringfinger_tip_x3D", "r_ringfinger1_x3D"],
        "right_pinky_tip": ["r_pinkyfinger_tip_x3D", "r_pinkyfinger1_x3D"],
        "right_thumb_tip": ["r_thumb_tip_x3D", "r_thumb1_x3D"],
        "left_index_tip": ["l_indexfinger_tip_x3D", "l_indexfinger1_x3D"],
        "left_middle_tip": ["l_middlefinger_tip_x3D", "l_middlefinger1_x3D"],
        "left_ring_tip": ["l_ringfinger_tip_x3D", "l_ringfinger1_x3D"],
        "left_pinky_tip": ["l_pinkyfinger_tip_x3D", "l_pinkyfinger1_x3D"],
        "left_thumb_tip": ["l_thumb_tip_x3D", "l_thumb1_x3D"],
    }
    x_col = resolve_column(df, [f"{name}_x3D"] + alias_map.get(name, []))
    if x_col is None:
        return np.array([]), np.array([]), np.array([])
    y_col = x_col.replace("_x3D", "_y3D")
    z_col = x_col.replace("_x3D", "_z3D")
    x = safe_array(df.get(x_col, pd.Series(dtype=float)))
    y = safe_array(df.get(y_col, pd.Series(dtype=float)))
    z = safe_array(df.get(z_col, pd.Series(dtype=float)))
    return x, y, z


def vector(a: tuple[float, float, float], b: tuple[float, float, float]) -> np.ndarray:
    return np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]], dtype=float)


def length(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def normalized(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return np.array([np.nan, np.nan, np.nan])
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.array([np.nan, np.nan, np.nan])


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    if np.isnan(a).any() or np.isnan(b).any() or a.size == 0 or b.size == 0:
        return float('nan')
    na = normalized(a)
    nb = normalized(b)
    dot = float(np.clip(np.dot(na, nb), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def periodicity_strength(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 6:
        return 0.0
    centered = values - np.mean(values)
    denom = float(np.dot(centered, centered))
    if denom <= 1e-12:
        return 0.0
    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    corr = corr / denom
    if len(corr) <= 2:
        return 0.0
    return float(np.nanmax(corr[2:]))


def motion_features(x: np.ndarray, y: np.ndarray, z: np.ndarray, prefix: str) -> dict:
    features = {
        f"{prefix}_count": int(len(x)),
        f"{prefix}_range_x": 0.0,
        f"{prefix}_range_y": 0.0,
        f"{prefix}_range_z": 0.0,
        f"{prefix}_vel_max": 0.0,
        f"{prefix}_vel_mean": 0.0,
        f"{prefix}_vel_std": 0.0,
        f"{prefix}_path_length": 0.0,
        f"{prefix}_accel_mean": 0.0,
        f"{prefix}_accel_std": 0.0,
        f"{prefix}_direction_change_mean": 0.0,
        f"{prefix}_periodicity": 0.0,
    }
    if len(x) >= 2:
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        velocities = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        features[f"{prefix}_range_x"] = float(np.nanmax(x) - np.nanmin(x))
        features[f"{prefix}_range_y"] = float(np.nanmax(y) - np.nanmin(y))
        features[f"{prefix}_range_z"] = float(np.nanmax(z) - np.nanmin(z))
        features[f"{prefix}_vel_max"] = float(np.nanmax(velocities))
        features[f"{prefix}_vel_mean"] = float(np.nanmean(velocities))
        features[f"{prefix}_vel_std"] = float(np.nanstd(velocities))
        features[f"{prefix}_path_length"] = float(np.nansum(velocities))
        features[f"{prefix}_periodicity"] = periodicity_strength(velocities)
        if len(velocities) >= 2:
            accel = np.abs(np.diff(velocities))
            features[f"{prefix}_accel_mean"] = float(np.nanmean(accel))
            features[f"{prefix}_accel_std"] = float(np.nanstd(accel))
        if len(x) >= 3:
            vectors = np.column_stack([dx, dy, dz])
            norms = np.linalg.norm(vectors, axis=1)
            valid = norms > 1e-12
            if np.count_nonzero(valid) >= 2:
                unit = vectors[valid] / norms[valid, None]
                turns = np.sum(unit[1:] * unit[:-1], axis=1)
                turns = np.clip(turns, -1.0, 1.0)
                features[f"{prefix}_direction_change_mean"] = float(np.nanmean(np.degrees(np.arccos(turns))))
    return features


def extract_skeleton_feature_set_from_df(df: pd.DataFrame) -> dict | None:
    if df is None or df.empty:
        return None
    features: dict[str, float] = {}
    if not has_usable_skeleton_data(df):
        return None
    parts = [
        "r_wrist",
        "r_elbow",
        "r_shoulder",
        "l_wrist",
        "l_elbow",
        "l_shoulder",
        "right_index_tip",
        "right_middle_tip",
        "right_ring_tip",
        "right_pinky_tip",
        "right_thumb_tip",
        "left_index_tip",
        "left_middle_tip",
        "left_ring_tip",
        "left_pinky_tip",
        "left_thumb_tip",
    ]
    for part in parts:
        x, y, z = get_coords(df, part)
        features.update(motion_features(x, y, z, part))

    point_names = [
        "r_wrist",
        "r_elbow",
        "r_shoulder",
        "l_wrist",
        "l_elbow",
        "l_shoulder",
        "right_index_tip",
        "right_thumb_tip",
    ]
    points = {}
    for name in point_names:
        x, y, z = get_coords(df, name)
        if len(x) > 0 and len(y) > 0 and len(z) > 0:
            points[name] = (float(x[-1]), float(y[-1]), float(z[-1]))
        else:
            points[name] = (float('nan'), float('nan'), float('nan'))

    features["dist_r_shoulder_elbow"] = distance(points["r_shoulder"], points["r_elbow"])
    features["dist_r_elbow_wrist"] = distance(points["r_elbow"], points["r_wrist"])
    features["dist_r_wrist_index"] = distance(points["r_wrist"], points["right_index_tip"])
    features["dist_r_wrist_thumb"] = distance(points["r_wrist"], points["right_thumb_tip"])
    features["dist_index_thumb"] = distance(points["right_index_tip"], points["right_thumb_tip"])

    bow_vec = vector(points["r_wrist"], points["right_index_tip"])
    thumb_vec = vector(points["r_wrist"], points["right_thumb_tip"])
    upperarm_vec = vector(points["r_shoulder"], points["r_elbow"])
    forearm_vec = vector(points["r_elbow"], points["r_wrist"])
    features["bow_length"] = float(length(bow_vec))
    features["bow_angle_elbow_wrist_index"] = float(angle_between(forearm_vec, bow_vec))
    features["bow_angle_shoulder_elbow_wrist"] = float(angle_between(upperarm_vec, forearm_vec))
    features["bow_angle_wrist_thumb_index"] = float(angle_between(thumb_vec, bow_vec))
    features["bow_thumb_distance"] = features["dist_index_thumb"]
    return features


def extract_skeleton_feature_set(csv_path: Path) -> dict | None:
    df = parse_mediapipe_csv(csv_path)
    return extract_skeleton_feature_set_from_df(df)


def frame_signal(y: np.ndarray, frame_length: int = 1024, hop_length: int = 512) -> np.ndarray:
    if y.size == 0:
        return np.empty((0, frame_length), dtype=float)
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))
    starts = np.arange(0, y.size - frame_length + 1, hop_length)
    if starts.size == 0:
        starts = np.array([0])
    indexes = starts[:, None] + np.arange(frame_length)[None, :]
    return y[indexes]


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(freqs: np.ndarray, n_mels: int = 20) -> np.ndarray:
    if freqs.size == 0:
        return np.empty((0, 0), dtype=float)
    mel_points = np.linspace(hz_to_mel(float(freqs[0])), hz_to_mel(float(freqs[-1])), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    filters = np.zeros((n_mels, freqs.size), dtype=float)
    for idx in range(n_mels):
        left, center, right = hz_points[idx], hz_points[idx + 1], hz_points[idx + 2]
        if center <= left or right <= center:
            continue
        rising = (freqs - left) / (center - left)
        falling = (right - freqs) / (right - center)
        filters[idx] = np.maximum(0.0, np.minimum(rising, falling))
    return filters


def safe_stat(values: np.ndarray, reducer, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    return float(reducer(values))


def extract_audio_features_from_array(y: np.ndarray, sr: int) -> dict | None:
    if y is None or y.size == 0 or sr <= 0:
        return None
    y = np.asarray(y, dtype=float)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak

    duration = float(len(y) / sr)
    frame_length = min(1024, max(256, int(sr * 0.046)))
    hop_length = max(128, frame_length // 2)
    frames = frame_signal(y, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

    nperseg = min(1024, max(256, len(y)))
    noverlap = min(nperseg // 2, nperseg - 1)
    freqs, _, stft = signal.stft(
        y,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(stft)
    power = magnitude ** 2
    mag_sum = np.sum(magnitude, axis=0) + 1e-12
    centroid = np.sum(freqs[:, None] * magnitude, axis=0) / mag_sum
    bandwidth = np.sqrt(np.sum(((freqs[:, None] - centroid[None, :]) ** 2) * magnitude, axis=0) / mag_sum)
    cumulative = np.cumsum(power, axis=0)
    thresholds = 0.85 * cumulative[-1:, :]
    rolloff_idx = np.argmax(cumulative >= thresholds, axis=0)
    rolloff = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)] if freqs.size else np.array([])

    if magnitude.shape[1] >= 2:
        flux = np.sqrt(np.sum(np.maximum(np.diff(magnitude, axis=1), 0.0) ** 2, axis=0))
    else:
        flux = np.array([], dtype=float)
    if flux.size:
        threshold = float(np.mean(flux) + 0.5 * np.std(flux))
        peaks, _ = signal.find_peaks(flux, height=threshold, distance=2)
    else:
        peaks = np.array([], dtype=int)
    onset_times = (peaks + 1) * (hop_length / sr)
    onset_intervals = np.diff(onset_times) if onset_times.size >= 2 else np.array([], dtype=float)

    filters = mel_filterbank(freqs, n_mels=20)
    if filters.size and power.size:
        mel_energy = filters @ power
        log_mel = np.log(mel_energy + 1e-10)
        mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:5]
    else:
        mfcc = np.zeros((5, 1), dtype=float)

    features = {
        "onset_count": int(len(peaks)),
        "onset_rate": float(len(peaks) / duration) if duration > 0 else 0.0,
        "onset_interval_mean": safe_stat(onset_intervals, np.mean),
        "onset_interval_std": safe_stat(onset_intervals, np.std),
        "onset_env_mean": safe_stat(flux, np.mean),
        "onset_env_std": safe_stat(flux, np.std),
        "onset_env_max": safe_stat(flux, np.max),
        "duration": duration,
        "centroid_mean": safe_stat(centroid, np.mean),
        "centroid_std": safe_stat(centroid, np.std),
        "centroid_delta_mean": safe_stat(np.abs(np.diff(centroid)), np.mean),
        "bandwidth_mean": safe_stat(bandwidth, np.mean),
        "bandwidth_std": safe_stat(bandwidth, np.std),
        "zcr_mean": safe_stat(zcr, np.mean),
        "zcr_std": safe_stat(zcr, np.std),
        "rms_mean": safe_stat(rms, np.mean),
        "rms_std": safe_stat(rms, np.std),
        "rms_delta_mean": safe_stat(np.abs(np.diff(rms)), np.mean),
        "rolloff_mean": safe_stat(rolloff, np.mean),
        "rolloff_std": safe_stat(rolloff, np.std),
    }
    for i in range(mfcc.shape[0]):
        features[f"mfcc_{i+1}_mean"] = safe_stat(mfcc[i], np.mean)
        features[f"mfcc_{i+1}_std"] = safe_stat(mfcc[i], np.std)
    return features


def extract_audio_features(wav_path: Path, max_duration: float = 20.0) -> dict | None:
    if not wav_path.exists():
        return None
    try:
        info = sf.info(str(wav_path))
        frames = int(min(info.frames, max_duration * info.samplerate)) if max_duration else info.frames
        y, sr = sf.read(str(wav_path), frames=frames, dtype="float32", always_2d=False)
    except Exception:
        return None
    return extract_audio_features_from_array(y, int(sr))


def find_matching_csv(wav_path: Path, base_dir: Path | None = None) -> Path | None:
    base = normalize_name(wav_path.stem)
    search_dirs = [wav_path.parent]
    if base_dir is not None:
        try:
            base_dir = base_dir.resolve()
        except Exception:
            pass
        if wav_path.is_absolute():
            try:
                relative = wav_path.relative_to(base_dir)
            except Exception:
                relative = None
            if relative is not None and len(relative.parts) >= 2:
                search_dirs = [wav_path.parent, *[base_dir / part for part in relative.parts[:-1]]]
        search_dirs.append(base_dir)
    else:
        search_dirs = [wav_path.parent]

    seen: set[Path] = set()
    for directory in search_dirs:
        if directory is None:
            continue
        try:
            directory = directory.resolve()
        except Exception:
            directory = Path(directory)
        if directory in seen:
            continue
        seen.add(directory)
        candidates = list(directory.glob("*_mediapipe*.csv"))
        for csv_path in candidates:
            if normalize_name(csv_path.stem).startswith(base):
                return csv_path

    for directory in search_dirs:
        if directory is None:
            continue
        try:
            directory = directory.resolve()
        except Exception:
            directory = Path(directory)
        if directory in seen:
            continue
        seen.add(directory)
        for parent in directory.parents:
            if parent in seen:
                continue
            seen.add(parent)
            candidates = list(parent.glob("*_mediapipe*.csv"))
            for csv_path in candidates:
                if normalize_name(csv_path.stem).startswith(base):
                    return csv_path

    if base_dir is not None:
        try:
            base_dir = base_dir.resolve()
        except Exception:
            pass
        for csv_path in sorted(base_dir.rglob("*_mediapipe*.csv")):
            if normalize_name(csv_path.stem).startswith(base):
                return csv_path

    for csv_path in sorted(Path(wav_path.root).rglob("*_mediapipe*.csv")):
        if normalize_name(csv_path.stem).startswith(base):
            return csv_path
    return None
