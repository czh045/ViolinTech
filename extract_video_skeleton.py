import argparse
import csv
import math
from pathlib import Path


try:
    import cv2
    import mediapipe as mp
except ImportError as exc:
    cv2 = None
    mp = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

mp_solutions = getattr(mp, "solutions", None) if mp is not None else None
mp_holistic = getattr(mp_solutions, "holistic", None) if mp_solutions is not None else None

LANDMARK_NAMES = [
    "r_wrist",
    "r_indexfinger_tip",
    "r_indexfinger_dip",
    "l_wrist",
    "l_indexfinger_tip",
    "l_indexfinger_dip",
]

POSE_LANDMARKS = {}
HAND_LANDMARKS = {}

if mp_holistic is not None:
    POSE_LANDMARKS = {
        "r_wrist": mp_holistic.PoseLandmark.RIGHT_WRIST,
        "l_wrist": mp_holistic.PoseLandmark.LEFT_WRIST,
    }
    HAND_LANDMARKS = {
        "r_indexfinger_tip": mp_holistic.HandLandmark.INDEX_FINGER_TIP,
        "r_indexfinger_dip": mp_holistic.HandLandmark.INDEX_FINGER_DIP,
        "l_indexfinger_tip": mp_holistic.HandLandmark.INDEX_FINGER_TIP,
        "l_indexfinger_dip": mp_holistic.HandLandmark.INDEX_FINGER_DIP,
    }


def require_dependencies() -> None:
    if cv2 is None:
        raise RuntimeError(f"OpenCV is not available: {IMPORT_ERROR}")
    if mp_holistic is None:
        detail = f": {IMPORT_ERROR}" if IMPORT_ERROR is not None else ""
        raise RuntimeError(f"MediaPipe Holistic is not available{detail}")


def landmark_value(landmark):
    if landmark is None:
        return (math.nan, math.nan, math.nan)
    return (
        float(landmark.x),
        float(landmark.y),
        float(landmark.z),
    )


def get_landmark(holistic_results, name):
    if name in POSE_LANDMARKS:
        idx = POSE_LANDMARKS[name]
        if holistic_results.pose_landmarks:
            return holistic_results.pose_landmarks.landmark[idx]
        return None
    if name in HAND_LANDMARKS:
        idx = HAND_LANDMARKS[name]
        if name.startswith("r_") and holistic_results.right_hand_landmarks:
            return holistic_results.right_hand_landmarks.landmark[idx]
        if name.startswith("l_") and holistic_results.left_hand_landmarks:
            return holistic_results.left_hand_landmarks.landmark[idx]
    return None


def process_video(video_path: Path, output_csv: Path) -> None:
    require_dependencies()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    header = ["frame", "time", "video_frame", "fps", "width", "height"]
    for name in LANDMARK_NAMES:
        header += [f"{name}_x3D", f"{name}_y3D", f"{name}_z3D"]

    rows = []
    detected_frames = 0
    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame_index += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                if results.pose_landmarks or results.right_hand_landmarks or results.left_hand_landmarks:
                    detected_frames += 1
                timestamp = frame_index / fps if fps > 0 else 0.0
                row = [frame_index, f"{timestamp:.4f}", frame_index, fps, width, height]
                for name in LANDMARK_NAMES:
                    lm = get_landmark(results, name)
                    row.extend(landmark_value(lm))
                rows.append(row)
    finally:
        capture.release()

    if not rows:
        raise RuntimeError(f"No frames read from video: {video_path}")
    if detected_frames == 0:
        raise RuntimeError(f"No pose or hand landmarks detected in video: {video_path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote skeleton CSV: {output_csv}")
    print(f"Frames: {len(rows)}, detected_frames: {detected_frames}, fps: {fps:.2f}, size: {width}x{height}")


def scan_videos(base_dir: Path, recursive: bool) -> list[Path]:
    return sorted(base_dir.rglob("*.mp4")) if recursive else sorted(base_dir.glob("*.mp4"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 3D MediaPipe skeleton data from videos")
    parser.add_argument("--base-dir", default=r"Z:\ViolinTech", help="Video root directory")
    parser.add_argument("--output-dir", default=None, help="Output CSV root; defaults to each video directory")
    parser.add_argument("--recursive", action="store_true", help="Search for mp4 files recursively")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    videos = scan_videos(base_dir, args.recursive)
    if not videos:
        raise SystemExit(f"No mp4 videos found under {base_dir}")

    for video_path in videos:
        out_base = Path(args.output_dir) / video_path.relative_to(base_dir).parent if args.output_dir else video_path.parent
        output_csv = out_base / f"{video_path.stem}_mediapipe.csv"
        process_video(video_path, output_csv)


if __name__ == "__main__":
    main()
