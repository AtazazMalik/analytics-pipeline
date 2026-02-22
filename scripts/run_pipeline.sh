#!/bin/bash
# ============================================================
#  Pipeline Entrypoint — runs inside Docker container
# ============================================================
VIDEO=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video) VIDEO="$2"; shift;;
        convert) exec python3 /app/convert_trt.py;;
        *) echo "Unknown arg: $1";;
    esac
    shift
done

if [ -z "$VIDEO" ]; then
    echo ""
    echo "Usage:"
    echo "  Run pipeline : docker run ... analytics-pipeline --video /data/video.mp4"
    echo "  Convert TRT  : docker run ... analytics-pipeline convert"
    exit 1
fi

echo ""
echo "============================================================"
echo "  Person Analytics Pipeline — TensorRT Edition"
echo "  Video: $VIDEO"
echo "============================================================"

[ ! -f "/app/models/yolov8n.engine" ] && \
    echo "⚠  TRT engines not found — using PyTorch (slower)" && \
    echo "   Run: docker run ... analytics-pipeline convert"

echo ""
echo "▶ Stage 1 — Detection & Tracking..."
python3 /app/stage1.py --video "$VIDEO" || { echo "❌ Stage 1 failed"; exit 1; }

echo ""
echo "▶ Stage 2 — Engagement Detection..."
python3 /app/stage2.py || { echo "❌ Stage 2 failed"; exit 1; }

echo ""
echo "▶ Stage 3 — Demographics Analysis..."
python3 /app/stage3.py || { echo "❌ Stage 3 failed"; exit 1; }

echo ""
echo "============================================================"
echo "  ✅ Pipeline Complete!"
echo "  Results  : /app/outputs/final_analytics.csv"
echo "  Preview  : /app/outputs/stage1_preview.mp4"
echo "============================================================"
