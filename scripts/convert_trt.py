#!/usr/bin/env python3
# ============================================================================
# TensorRT Engine Converter
# Run ONCE inside Docker on Jetson Nano before running the pipeline
# Converts yolov8n.pt and yolov8n-pose.pt → fast .engine files
#
# Usage:
#   docker run --runtime nvidia --rm \
#     -v /media/usb/models:/app/models \
#     analytics-pipeline python3 /app/convert_trt.py
# ============================================================================
import os, time, torch
from ultralytics import YOLO

MODELS_DIR   = "/app/models"
IMGSZ        = 416      # Must match TARGET_W/H in stage1.py
WORKSPACE_GB = 1        # Keep at 1GB — Jetson Nano only has 4GB shared
HALF         = True     # FP16 = faster + less VRAM

MODELS = [
    ("yolov8n",      "yolov8n.pt",      "yolov8n.engine"),
    ("yolov8n-pose", "yolov8n-pose.pt", "yolov8n-pose.engine"),
]

def convert(display_name, pt_name, engine_name):
    engine_path = os.path.join(MODELS_DIR, engine_name)
    pt_path     = os.path.join(MODELS_DIR, pt_name)

    print("\n" + "="*60)
    print("Converting: " + display_name)
    print("="*60)

    if os.path.exists(engine_path):
        print("Engine already exists — skipping.")
        print("Delete it to force re-conversion: " + engine_path)
        return True

    print("Settings:")
    print("  Image size  : {}x{}".format(IMGSZ,IMGSZ))
    print("  FP16 (half) : {}".format(HALF))
    print("  Workspace   : {} GB".format(WORKSPACE_GB))
    print("  This takes 5-15 minutes. Please wait...")

    t0 = time.time()
    try:
        model = YOLO(pt_path)   # downloads if not present
        model.export(
            format="engine",
            imgsz=IMGSZ,
            half=HALF,
            workspace=WORKSPACE_GB,
            verbose=False,
            device=0
        )
        # Move engine to models dir if saved elsewhere
        default = pt_path.replace(".pt",".engine")
        if os.path.exists(default) and default != engine_path:
            os.rename(default, engine_path)

        print("✅ Done in {:.1f}s — saved to: {}".format(time.time()-t0, engine_path))
        return True
    except Exception as e:
        print("❌ Failed: {}".format(e))
        print("   Pipeline will use PyTorch fallback instead.")
        return False

def main():
    print("============================================================")
    print("  TensorRT Engine Converter")
    print("  CUDA available: " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("  GPU: " + torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("  GPU RAM: {:.1f} GB".format(mem))
    print("============================================================")

    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available!")
        print("Make sure you started Docker with --runtime nvidia")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    results = []
    for display_name, pt_name, engine_name in MODELS:
        ok = convert(display_name, pt_name, engine_name)
        results.append((display_name, ok))

    print("\n============================================================")
    print("  Conversion Summary")
    print("============================================================")
    all_ok = True
    for name, ok in results:
        status = "OK" if ok else "FAILED (PyTorch fallback will be used)"
        print("  {:<20}: {}".format(name, status))
        if not ok: all_ok = False

    if all_ok:
        print("\n✅ Both engines ready! Expected speedup: 3-5x vs PyTorch")
    else:
        print("\n⚠️  Some conversions failed — pipeline still works via PyTorch")

    print("\n→ Now run the pipeline with your video!")

if __name__ == "__main__":
    main()
