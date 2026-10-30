"""
Evaluate the repository's MiniFASNet presentation-attack-detection (PAD) pipeline.

Dataset layout (create this locally; do not commit private biometric data):
    data/pad_eval/
        real/   # genuine/live-face images
        spoof/  # presentation-attack images: printed photos, screen replays, etc.

Run from the repository root:
    python evaluation/evaluate_pad.py --data-dir data/pad_eval

The script uses the same PAD fusion pipeline as app/pad_engine.py and InsightFace
for face detection. If no face (or not exactly one face) is detected, the sample
is recorded as SPOOF/rejected; this counts as a false rejection for a real image
and a correct rejection for a spoof image. Review these cases in the CSV.

Important:
- Use a held-out test set not used to tune thresholds or train models.
- Include diverse subjects, capture devices, lighting, and attack types.
- The official MiniFASNet outputs are fused by app/pad_engine.py; class 1 is REAL,
  while classes 0 and 2 are treated as SPOOF.
- FAR here means attack acceptance rate: spoof samples predicted REAL / all spoof
  samples. FRR means genuine rejection rate: real samples predicted SPOOF / all
  real samples. These definitions should be stated when reporting results.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MiniFASNet PAD on labeled real/spoof image folders."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pad_eval",
        help="Folder containing real/ and spoof/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results",
        help="Where to write the per-image CSV and text report.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help="PAD engine currently defaults to CPU in this project.",
    )
    return parser.parse_args()


def image_paths(folder: Path):
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    args = get_args()
    real_dir = args.data_dir / "real"
    spoof_dir = args.data_dir / "spoof"

    if not real_dir.is_dir() or not spoof_dir.is_dir():
        raise SystemExit(
            "Dataset folders not found. Expected:\n"
            f"  {real_dir}\n  {spoof_dir}\n"
            "Create these folders and place labeled evaluation images inside."
        )

    samples = [
        *((path, "REAL") for path in image_paths(real_dir)),
        *((path, "SPOOF") for path in image_paths(spoof_dir)),
    ]
    if not samples:
        raise SystemExit(f"No supported images found under {args.data_dir}")

    # Import after checking the dataset paths so setup errors are easier to read.
    from app.face_engine import FaceEngine
    from app.pad_engine import PAD

    print(f"Found {sum(label == 'REAL' for _, label in samples)} real images.")
    print(f"Found {sum(label == 'SPOOF' for _, label in samples)} spoof images.")
    print("Loading face detector and PAD models...")
    face_engine = FaceEngine()
    pad = PAD(device=args.device)

    y_true = []
    y_pred = []
    rows = []

    for index, (path, true_label) in enumerate(samples, start=1):
        image = cv2.imread(str(path))
        reason = ""
        confidence = 0.0
        class_0 = class_1 = class_2 = 0.0

        if image is None:
            print(f"[{index}/{len(samples)}] UNREADABLE: {path}")
            rows.append({
                "path": str(path.relative_to(args.data_dir)),
                "true_label": true_label,
                "predicted_label": "ERROR",
                "confidence": "",
                "class_0_spoof_score": "",
                "class_1_real_score": "",
                "class_2_spoof_score": "",
                "status": "unreadable_image",
            })
            continue

        faces = face_engine.detect_faces(image)
        if len(faces) != 1:
            predicted_label = "SPOOF"
            reason = f"rejected_no_single_face_detected (faces={len(faces)})"
        else:
            bbox_xyxy = np.asarray(faces[0].bbox, dtype=float)
            x1, y1, x2, y2 = bbox_xyxy.tolist()
            bbox_xywh = [
                int(round(x1)),
                int(round(y1)),
                max(1, int(round(x2 - x1))),
                max(1, int(round(y2 - y1))),
            ]
            result = pad.check(image, bbox_xywh)
            predicted_label = result["label"]
            confidence = float(result["confidence"])
            class_0 = float(result["class_0"])
            class_1 = float(result["class_1"])
            class_2 = float(result["class_2"])
            reason = "ok"

        y_true.append(true_label)
        y_pred.append(predicted_label)
        rows.append({
            "path": str(path.relative_to(args.data_dir)),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "class_0_spoof_score": class_0,
            "class_1_real_score": class_1,
            "class_2_spoof_score": class_2,
            "status": reason,
        })
        print(f"[{index}/{len(samples)}] true={true_label:5s} pred={predicted_label:5s} {path.name}")

    # Unreadable images are excluded from metrics but retained in the CSV.
    valid_count = len(y_true)
    if valid_count == 0:
        raise SystemExit("No readable samples were evaluated.")

    labels = ["REAL", "SPOOF"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Rows = actual, columns = predicted; order is REAL, SPOOF.
    real_correct, real_rejected = int(cm[0, 0]), int(cm[0, 1])
    spoof_accepted, spoof_rejected = int(cm[1, 0]), int(cm[1, 1])

    accuracy = accuracy_score(y_true, y_pred)
    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    # Index 1 corresponds to SPOOF because labels = ["REAL", "SPOOF"].
    precision, recall, f1 = float(precisions[1]), float(recalls[1]), float(f1_scores[1])
    real_count = int(sum(label == "REAL" for label in y_true))
    spoof_count = int(sum(label == "SPOOF" for label in y_true))
    frr = (real_rejected / real_count) if real_count else float("nan")
    far = (spoof_accepted / spoof_count) if spoof_count else float("nan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "pad_predictions.csv"
    report_path = args.output_dir / "pad_evaluation_report.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "MINIFASNET PAD EVALUATION",
        "=" * 32,
        f"Dataset directory: {args.data_dir.resolve()}",
        f"Readable evaluated samples: {valid_count}",
        f"Unreadable images (excluded from metrics): {sum(row['status'] == 'unreadable_image' for row in rows)}",
        f"REAL samples: {real_count}",
        f"SPOOF samples: {spoof_count}",
        "",
        "Confusion matrix (rows = actual, columns = predicted; order REAL, SPOOF):",
        "                 Pred REAL   Pred SPOOF",
        f"Actual REAL      {cm[0, 0]:9d}   {cm[0, 1]:10d}",
        f"Actual SPOOF     {cm[1, 0]:9d}   {cm[1, 1]:10d}",
        "",
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)",
        f"SPOOF precision: {precision:.4f}",
        f"SPOOF recall: {recall:.4f}",
        f"SPOOF F1-score: {f1:.4f}",
        f"FAR (spoof incorrectly accepted as REAL): {far:.4f} ({far * 100:.2f}%)",
        f"FRR (real incorrectly rejected as SPOOF): {frr:.4f} ({frr * 100:.2f}%)",
        "",
        "Per-class classification report:",
        classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "Notes:",
        "- No-face/multiple-face images are treated as rejected (SPOOF).",
        "- FAR/FRR are calculated from the labeled samples successfully read.",
        "- These metrics describe only this evaluation dataset and capture setup.",
        "- For biometric PAD benchmarking, report attack types and use a subject-disjoint, held-out test set.",
    ]
    report_text = "\n".join(report) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"Per-image predictions: {csv_path}")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
