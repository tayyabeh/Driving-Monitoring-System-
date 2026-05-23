import torch          # must be first on Windows
import os
from pathlib import Path
from ultralytics import YOLO

DATA_YAML  = "data.yaml"
WEIGHTS    = "best.pt"
IMG_SZ     = 512
CONF       = 0.25
IOU        = 0.60
DEVICE     = "cpu"

print("=" * 60)
print("  Driver Distraction Detection — Validate & Test")
print("=" * 60)

# ── Load model ──────────────────────────────────────────────
model = YOLO(WEIGHTS)
print(f"\n✅ Model loaded : {WEIGHTS}")
print(f"   Classes      : {list(model.names.values())}\n")

# ════════════════════════════════════════════════════════════
# 1.  VALIDATION  (val split)
# ════════════════════════════════════════════════════════════
print("─" * 60)
print("  STEP 1 — Validation on val/ split")
print("─" * 60)

val_results = model.val(
    data    = DATA_YAML,
    split   = "val",
    imgsz   = IMG_SZ,
    conf    = CONF,
    iou     = IOU,
    device  = DEVICE,
    project = "runs/val",
    name    = "val_run",
    plots   = True,        # confusion matrix, PR curve, F1 curve
    verbose = True,
)

print("\n📊  Validation Metrics:")
print(f"   mAP@50       : {val_results.box.map50:.4f}")
print(f"   mAP@50-95    : {val_results.box.map:.4f}")
print(f"   Precision    : {val_results.box.mp:.4f}")
print(f"   Recall       : {val_results.box.mr:.4f}")

# Per-class table
print("\n   Per-class AP@50:")
class_names = list(model.names.values())
maps = val_results.box.ap50          # array of AP50 per class
for name, ap in zip(class_names, maps):
    bar = "█" * int(ap * 20)
    print(f"   {name:<35} {ap:.3f}  {bar}")

print(f"\n   📁 Plots saved → runs/val/val_run/")

# ════════════════════════════════════════════════════════════
# 2.  TESTING  (test split)
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  STEP 2 — Testing on test/ split")
print("─" * 60)

test_results = model.val(
    data    = DATA_YAML,
    split   = "test",
    imgsz   = IMG_SZ,
    conf    = CONF,
    iou     = IOU,
    device  = DEVICE,
    project = "runs/test",
    name    = "test_run",
    plots   = True,
    verbose = True,
)

print("\n📊  Test Metrics:")
print(f"   mAP@50       : {test_results.box.map50:.4f}")
print(f"   mAP@50-95    : {test_results.box.map:.4f}")
print(f"   Precision    : {test_results.box.mp:.4f}")
print(f"   Recall       : {test_results.box.mr:.4f}")

print("\n   Per-class AP@50:")
maps_test = test_results.box.ap50
for name, ap in zip(class_names, maps_test):
    bar = "█" * int(ap * 20)
    print(f"   {name:<35} {ap:.3f}  {bar}")

print(f"\n   📁 Plots saved → runs/test/test_run/")

# ════════════════════════════════════════════════════════════
# 3.  SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  {'Metric':<20} {'Val':>10} {'Test':>10}")
print(f"  {'-'*40}")
print(f"  {'mAP@50':<20} {val_results.box.map50:>10.4f} {test_results.box.map50:>10.4f}")
print(f"  {'mAP@50-95':<20} {val_results.box.map:>10.4f} {test_results.box.map:>10.4f}")
print(f"  {'Precision':<20} {val_results.box.mp:>10.4f} {test_results.box.mp:>10.4f}")
print(f"  {'Recall':<20} {val_results.box.mr:>10.4f} {test_results.box.mr:>10.4f}")
print("=" * 60)
print("\n✅ Done! Open the folders below to view plots:")
print("   runs/val/val_run/   — validation confusion matrix, PR curve")
print("   runs/test/test_run/ — test confusion matrix, PR curve")
