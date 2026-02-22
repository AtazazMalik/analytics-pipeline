#!/usr/bin/env python3
# ============================================================================
# STAGE 1 — Person Detection, Tracking, ROI Filtering, Crop Saving
# Jetson Nano | TensorRT accelerated | Python 3.6
# ============================================================================
import cv2, os, json, time, torch, argparse
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default="/data/input.mp4")
args = parser.parse_args()

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_VIDEO          = args.video
OUTPUT_DIR           = "/app/stage1_output"
METADATA_PATH        = "/app/stage1_metadata.json"
MODELS_DIR           = "/app/models"

FRAME_SKIP           = 4
TARGET_W             = 416
TARGET_H             = 416      # Square — required for TRT engine
CONF_THRESH          = 0.4
ROI_MARGIN_X         = 0.15
ROI_MARGIN_Y         = 0.15
TRACK_CLASSES        = [0, 1, 2, 3, 5, 7]
PERSON_CLASS         = 0
MAX_CROPS_PER_PERSON = 300

# ── Model loader: TRT engine first, PyTorch fallback ─────────────────────────
def load_model(name):
    engine = os.path.join(MODELS_DIR, name + ".engine")
    if os.path.exists(engine):
        print("  [TRT] Loading engine: " + engine)
        return YOLO(engine), "TensorRT"
    print("  [PT]  Engine not found — using PyTorch (.pt)")
    print("        Run convert_trt.py first for 3-5x speedup!")
    m = YOLO(name + ".pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev)
    if dev == "cuda":
        m.model = m.model.half()
    return m, "PyTorch"

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_roi(w, h):
    return (int(w*ROI_MARGIN_X), int(h*ROI_MARGIN_Y),
            int(w*(1-ROI_MARGIN_X)), int(h*(1-ROI_MARGIN_Y)))

def center_in_roi(box, roi):
    cx = (box[0]+box[2])/2.0
    cy = (box[1]+box[3])/2.0
    return roi[0]<=cx<=roi[2] and roi[1]<=cy<=roi[3]

def draw_roi(frame, roi):
    cv2.rectangle(frame, (roi[0],roi[1]), (roi[2],roi[3]), (0,255,0), 2)
    cv2.putText(frame,"ROI",(roi[0]+5,roi[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

def save_crop(frame, box, frame_num, person_dir, count):
    if count >= MAX_CROPS_PER_PERSON:
        return
    x1,y1,x2,y2 = map(int,box)
    x1=max(0,x1); y1=max(0,y1)
    x2=min(frame.shape[1],x2); y2=min(frame.shape[0],y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    cv2.imwrite(os.path.join(person_dir,"frame_{:06d}.jpg".format(frame_num)),
                crop, [cv2.IMWRITE_JPEG_QUALITY,80])

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("="*60)
    print("STAGE 1 — Detection + Tracking + Crop Saving")
    print("CUDA: " + str(torch.cuda.is_available()))
    print("="*60)

    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError("Video not found: " + INPUT_VIDEO)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    model, backend = load_model("yolov8n")
    print("Backend: " + backend)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("Cannot open: " + INPUT_VIDEO)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video: {}x{} @ {:.1f}fps | {} frames".format(orig_w,orig_h,fps,total))
    print("Processing at {}x{} every {}th frame".format(TARGET_W,TARGET_H,FRAME_SKIP))

    roi = compute_roi(TARGET_W, TARGET_H)
    print("ROI: " + str(roi))

    os.makedirs("/app/outputs", exist_ok=True)
    eff_fps   = max(1, int(fps/FRAME_SKIP))
    preview   = cv2.VideoWriter("/app/outputs/stage1_preview.mp4",
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                eff_fps, (TARGET_W,TARGET_H))

    metadata    = {}
    frame_count = 0
    t0          = time.time()
    fps_buf     = []

    print("Processing frames...")
    while cap.isOpened():
        if frame_count % FRAME_SKIP != 0:
            cap.grab(); frame_count += 1; continue

        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        ti = time.time()

        try:
            results = model.track(frame, persist=True, verbose=False,
                                  conf=CONF_THRESH, classes=TRACK_CLASSES,
                                  tracker="bytetrack.yaml")
        except Exception as e:
            print("  [WARN] frame {}: {}".format(frame_count,e))
            frame_count += 1; continue

        fps_buf.append(1.0/(time.time()-ti+1e-9))
        if len(fps_buf)>30: fps_buf.pop(0)

        draw_roi(frame, roi)

        for r in results:
            if r.boxes.id is None: continue
            ids     = r.boxes.id.int().cpu().numpy()
            boxes   = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()

            for i, oid in enumerate(ids):
                oid  = int(oid)
                box  = boxes[i]
                cls  = int(classes[i])
                inr  = center_in_roi(box, roi)

                x1,y1,x2,y2 = map(int,box)
                col = (0,255,0) if inr else (128,128,128)
                cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                lbl = "{}{}".format("P" if cls==PERSON_CLASS else "V", oid)
                cv2.putText(frame,lbl,(x1,max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)

                if cls != PERSON_CLASS or not inr: continue

                if oid not in metadata:
                    pdir = os.path.join(OUTPUT_DIR,"ID_{}".format(oid))
                    os.makedirs(pdir, exist_ok=True)
                    metadata[oid] = {"entry_time_sec": frame_count/fps,
                                     "frames":[], "roi_frame_count":0,
                                     "crop_count":0, "person_dir":pdir}
                    print("  New ID {} @ frame {}".format(oid,frame_count))

                metadata[oid]["frames"].append(frame_count)
                metadata[oid]["roi_frame_count"] += 1
                save_crop(frame,box,frame_count,
                          metadata[oid]["person_dir"],metadata[oid]["crop_count"])
                metadata[oid]["crop_count"] += 1

        avg_fps = sum(fps_buf)/len(fps_buf) if fps_buf else 0
        cv2.putText(frame,"F:{} {:.1f}fps [{}]".format(frame_count,avg_fps,backend[:3]),
                    (5,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
        preview.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print("  Frame {}/{} | IDs:{} | {:.1f}fps | {:.0f}s elapsed".format(
                frame_count,total,len(metadata),avg_fps,time.time()-t0))

    cap.release(); preview.release()

    clean = {}
    for pid,data in metadata.items():
        roi_t = (data["roi_frame_count"]*FRAME_SKIP)/fps
        clean[str(pid)] = {
            "entry_time_sec": round(data["entry_time_sec"],2),
            "roi_time_sec":   round(roi_t,2),
            "frame_count":    data["roi_frame_count"],
            "frames_list":    data["frames"],
            "person_dir":     data["person_dir"]
        }

    with open(METADATA_PATH,"w") as f:
        json.dump(clean,f,indent=2)

    elapsed = time.time()-t0
    avg_fps = sum(fps_buf)/len(fps_buf) if fps_buf else 0
    print("\n✅ Stage 1 Complete! ({:.1f}s)".format(elapsed))
    print("   Backend         : " + backend)
    print("   Avg inference   : {:.1f} FPS".format(avg_fps))
    print("   Persons tracked : {}".format(len(clean)))
    print("   Crops saved to  : " + OUTPUT_DIR)
    print("   Preview video   : /app/outputs/stage1_preview.mp4")

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("   GPU memory released")

if __name__ == "__main__":
    run()
