from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RESULTS_PATH = Path("tmp_out") / "paper_experiment_results.json"
IMPORTANCE_PATH = Path("tmp_out") / "paper_feature_importance.csv"
CONFUSION_OUT = Path("confusion_matrix.png")
IMPORTANCE_OUT = Path("feature_importance.png")
FRAMEWORK_OUT = Path("system_framework.png")
DISPLAY_LABELS = {
    "double_stop": "Double stops",
    "dungong": "Stopped bowing",
    "pizzicato": "Pizzicato",
    "plain": "Plain playing",
    "spiccato": "Spiccato",
    "vibrato": "Vibrato",
}


def pretty_feature_name(name: str) -> str:
    replacements = {
        "right_": "R ",
        "left_": "L ",
        "r_": "R ",
        "l_": "L ",
        "dist_": "Dist ",
        "bow_": "Bow ",
        "direction_change_mean": "turn mean",
        "vel_mean": "velocity mean",
        "vel_std": "velocity std",
        "vel_max": "velocity max",
        "accel_mean": "accel. mean",
        "accel_std": "accel. std",
        "centroid_delta_mean": "centroid delta mean",
        "onset_env_std": "onset env. std",
        "rms_mean": "RMS mean",
    }
    label = name
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.replace("_", " ").strip().title().replace("Mfcc", "MFCC").replace("Rms", "RMS")


def plot_confusion(results: dict) -> None:
    labels = results["class_names"]
    display_labels = [DISPLAY_LABELS.get(label, label) for label in labels]
    modality_results = results.get("grouped_modalities", results["modalities"])
    matrix = np.array(modality_results["multimodal"]["confusion_matrix"], dtype=int)

    plt.figure(figsize=(7.2, 5.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_OUT, dpi=300, bbox_inches="tight")
    plt.close()


def plot_importance() -> None:
    df = pd.read_csv(IMPORTANCE_PATH).head(15).iloc[::-1].copy()
    df["label"] = df["feature"].map(pretty_feature_name)
    colors = [
        "#3b82f6"
        if not name.startswith(("right_", "left_", "r_", "l_", "dist_", "bow_"))
        else "#14b8a6"
        for name in df["feature"]
    ]

    plt.figure(figsize=(7.2, 5.8))
    plt.barh(df["label"], df["importance"], color=colors)
    plt.xlabel("Random Forest impurity importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_OUT, dpi=300, bbox_inches="tight")
    plt.close()


def _box(ax, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10, color="#111827")


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#374151", "shrinkA": 3, "shrinkB": 3},
    )


def plot_framework() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.03, 0.40), 0.13, 0.20, "Input video\nclip", "#dbeafe")
    _box(ax, (0.24, 0.64), 0.18, 0.18, "Audio stream\nWAV extraction", "#e0f2fe")
    _box(ax, (0.24, 0.18), 0.18, 0.18, "Visual stream\nMediaPipe Holistic", "#dcfce7")
    _box(ax, (0.48, 0.64), 0.19, 0.18, "Audio descriptors\nonset, spectral,\nRMS, MFCC", "#e0f2fe")
    _box(ax, (0.48, 0.18), 0.19, 0.18, "Skeleton descriptors\nmotion and bowing\ngeometry", "#dcfce7")
    _box(ax, (0.72, 0.40), 0.13, 0.20, "Feature fusion\n198-D vector", "#fef3c7")
    _box(ax, (0.89, 0.40), 0.08, 0.20, "Random\nForest", "#fde68a")
    _box(ax, (0.89, 0.08), 0.08, 0.18, "Technique\nlabel", "#fee2e2")

    _arrow(ax, (0.16, 0.52), (0.24, 0.73))
    _arrow(ax, (0.16, 0.48), (0.24, 0.27))
    _arrow(ax, (0.42, 0.73), (0.48, 0.73))
    _arrow(ax, (0.42, 0.27), (0.48, 0.27))
    _arrow(ax, (0.67, 0.73), (0.72, 0.54))
    _arrow(ax, (0.67, 0.27), (0.72, 0.46))
    _arrow(ax, (0.85, 0.50), (0.89, 0.50))
    _arrow(ax, (0.93, 0.40), (0.93, 0.26))

    ax.text(
        0.48,
        0.03,
        "Duration, raw count, and path-length descriptors are excluded before training.",
        ha="center",
        va="center",
        fontsize=9,
        color="#4b5563",
    )
    plt.tight_layout()
    plt.savefig(FRAMEWORK_OUT, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    plot_framework()
    plot_confusion(results)
    plot_importance()
    print(f"Saved {FRAMEWORK_OUT}, {CONFUSION_OUT}, and {IMPORTANCE_OUT}")


if __name__ == "__main__":
    main()
