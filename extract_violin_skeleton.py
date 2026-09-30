import argparse
import csv
import math
import os
from pathlib import Path


os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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

if mp_holistic is None:
    POSE_POINTS = {}
    HAND_POINTS = {}
else:
    POSE_POINTS = {
        "r_shoulder": mp_holistic.PoseLandmark.RIGHT_SHOULDER,
        "r_elbow": mp_holistic.PoseLandmark.RIGHT_ELBOW,
        "r_wrist": mp_holistic.PoseLandmark.RIGHT_WRIST,
        "l_shoulder": mp_holistic.PoseLandmark.LEFT_SHOULDER,
        "l_elbow": mp_holistic.PoseLandmark.LEFT_ELBOW,
        "l_wrist": mp_holistic.PoseLandmark.LEFT_WRIST,
    }

    HAND_POINTS = {
        "thumb_cmc": mp_holistic.HandLandmark.THUMB_CMC,
        "thumb_mcp": mp_holistic.HandLandmark.THUMB_MCP,
        "thumb_ip": mp_holistic.HandLandmark.THUMB_IP,
        "thumb_tip": mp_holistic.HandLandmark.THUMB_TIP,
        "index_mcp": mp_holistic.HandLandmark.INDEX_FINGER_MCP,
        "index_pip": mp_holistic.HandLandmark.INDEX_FINGER_PIP,
        "index_dip": mp_holistic.HandLandmark.INDEX_FINGER_DIP,
        "index_tip": mp_holistic.HandLandmark.INDEX_FINGER_TIP,
        "middle_mcp": mp_holistic.HandLandmark.MIDDLE_FINGER_MCP,
        "middle_pip": mp_holistic.HandLandmark.MIDDLE_FINGER_PIP,
        "middle_dip": mp_holistic.HandLandmark.MIDDLE_FINGER_DIP,
        "middle_tip": mp_holistic.HandLandmark.MIDDLE_FINGER_TIP,
        "ring_mcp": mp_holistic.HandLandmark.RING_FINGER_MCP,
        "ring_pip": mp_holistic.HandLandmark.RING_FINGER_PIP,
        "ring_dip": mp_holistic.HandLandmark.RING_FINGER_DIP,
        "ring_tip": mp_holistic.HandLandmark.RING_FINGER_TIP,
        "pinky_mcp": mp_holistic.HandLandmark.PINKY_MCP,
        "pinky_pip": mp_holistic.HandLandmark.PINKY_PIP,
        "pinky_dip": mp_holistic.HandLandmark.PINKY_DIP,
        "pinky_tip": mp_holistic.HandLandmark.PINKY_TIP,
        "wrist": mp_holistic.HandLandmark.WRIST,
    }


def require_dependencies() -> None:
    if cv2 is None:
        raise RuntimeError(f"OpenCV is not available: {IMPORT_ERROR}")
    if mp_holistic is None:
        detail = f": {IMPORT_ERROR}" if IMPORT_ERROR is not None else ""
        raise RuntimeError(f"MediaPipe Holistic is not available{detail}")


def landmark_to_tuple(landmark):
    if landmark is None:
        return (math.nan, math.nan, math.nan)
    return float(landmark.x), float(landmark.y), float(landmark.z)


def pose_landmark(results, key):
    if results.pose_landmarks:
        return results.pose_landmarks.landmark[POSE_POINTS[key]]
    return None


def hand_landmark(results, hand_side, key):
    hand = results.right_hand_landmarks if hand_side == "right" else results.left_hand_landmarks
    if hand:
        return hand.landmark[HAND_POINTS[key]]
    return None


def vector_from(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def vector_length(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalized(v):
    length = vector_length(v)
    if length == 0 or math.isnan(length):
        return (math.nan, math.nan, math.nan)
    return (v[0] / length, v[1] / length, v[2] / length)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def angle_between(a, b):
    na = normalized(a)
    nb = normalized(b)
    if any(math.isnan(x) for x in (*na, *nb)):
        return math.nan
    cosine = dot(na, nb)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def build_header():
    header = ["frame", "time", "video_frame", "fps", "width", "height"]
    for name in POSE_POINTS:
        header += [f"{name}_x3D", f"{name}_y3D", f"{name}_z3D", f"{name}_visibility"]
    for side in ("right", "left"):
        for name in HAND_POINTS:
            header += [f"{side}_{name}_x3D", f"{side}_{name}_y3D", f"{side}_{name}_z3D"]
    header += [
        "bow_wrist_to_index_x",
        "bow_wrist_to_index_y",
        "bow_wrist_to_index_z",
        "bow_length",
        "bow_thumb_spread",
        "bow_angle_elbow_wrist_index",
        "bow_angle_shoulder_elbow_wrist",
        "bow_angle_wrist_thumb_index",
    ]
    return header


def extract_row(results, frame_index, timestamp, fps, width, height):
    row = [frame_index, f"{timestamp:.4f}", frame_index, fps, width, height]
    pose_points = {}
    for name in POSE_POINTS:
        lm = pose_landmark(results, name)
        x, y, z = landmark_to_tuple(lm)
        visibility = float(lm.visibility) if lm is not None and hasattr(lm, "visibility") else math.nan
        row += [x, y, z, visibility]
        pose_points[name] = (x, y, z)

    hand_points = {}
    for side in ("right", "left"):
        for name in HAND_POINTS:
            lm = hand_landmark(results, side, name)
            x, y, z = landmark_to_tuple(lm)
            row += [x, y, z]
            hand_points[f"{side}_{name}"] = (x, y, z)

    r_wrist = pose_points.get("r_wrist", (math.nan, math.nan, math.nan))
    r_elbow = pose_points.get("r_elbow", (math.nan, math.nan, math.nan))
    r_shoulder = pose_points.get("r_shoulder", (math.nan, math.nan, math.nan))
    r_index_tip = hand_points.get("right_index_tip", (math.nan, math.nan, math.nan))
    r_thumb_tip = hand_points.get("right_thumb_tip", (math.nan, math.nan, math.nan))

    bow_vec = vector_from(r_wrist, r_index_tip)
    thumb_vec = vector_from(r_wrist, r_thumb_tip)
    bow_len = vector_length(bow_vec)
    bow_thumb_spread = vector_length(vector_from(r_index_tip, r_thumb_tip))
    elbow_wrist_index = angle_between(vector_from(r_elbow, r_wrist), bow_vec)
    shoulder_elbow_wrist = angle_between(vector_from(r_shoulder, r_elbow), vector_from(r_elbow, r_wrist))
    wrist_thumb_index = angle_between(thumb_vec, bow_vec)

    row += [
        bow_vec[0],
        bow_vec[1],
        bow_vec[2],
        bow_len,
        bow_thumb_spread,
        elbow_wrist_index,
        shoulder_elbow_wrist,
        wrist_thumb_index,
    ]
    return row


def process_video(video_path: Path, output_csv: Path) -> None:
    require_dependencies()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

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
                rows.append(extract_row(results, frame_index, timestamp, fps, width, height))
    finally:
        capture.release()

    if not rows:
        raise RuntimeError(f"No frames read from video: {video_path}")
    if detected_frames == 0:
        raise RuntimeError(f"No pose or hand landmarks detected in video: {video_path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(build_header())
        writer.writerows(rows)

    print(f"Wrote skeleton CSV: {output_csv}")
    print(f"Frames: {len(rows)}, detected_frames: {detected_frames}, fps: {fps:.2f}, size: {width}x{height}")


def scan_videos(base_dir: Path, recursive: bool):
    return sorted(base_dir.rglob("*.mp4")) if recursive else sorted(base_dir.glob("*.mp4"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract violin pose and hand skeleton data from videos")
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
        output_csv = out_base / f"{video_path.stem}_violin_skeleton.csv"
        process_video(video_path, output_csv)


if __name__ == "__main__":
    main()
