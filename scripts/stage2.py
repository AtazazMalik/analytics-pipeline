#!/usr/bin/env python3
# ============================================================================
# STAGE 2 — Pose Estimation + Engagement Detection
# Jetson Nano | TensorRT accelerated | Python 3.6
# ============================================================================
import cv2, os, json, time, torch
import numpy as np
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
STAGE1_METADATA   = "/app/stage1_metadata.json"
STAGE2_OUTPUT     = "/app/stage2_engagement.json"
MODELS_DIR        = "/app/models"

LOOK_THRESHOLD_SEC = 4.0
# Set FPS_EQUIVALENT = your_video_fps / FRAME_SKIP used in stage1
# e.g. 30fps video / FRAME_SKIP 4 = 7.5
FPS_EQUIVALENT     = 7.5
FRAMES_REQUIRED    = int(LOOK_THRESHOLD_SEC * FPS_EQUIVALENT)
POSE_CONF          = 0.4

# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(name):
    engine = os.path.join(MODELS_DIR, name + ".engine")
    if os.path.exists(engine):
        print("  [TRT] Loading engine: " + engine)
        return YOLO(engine), "TensorRT"
    print("  [PT]  Engine not found — using PyTorch (.pt)")
    m = YOLO(name + ".pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev)
    if dev == "cuda":
        m.model = m.model.half()
    return m, "PyTorch"

# ── Looking detection ─────────────────────────────────────────────────────────
def is_looking(keypoints):
    nose  = keypoints[0]
    l_eye = keypoints[1]
    r_eye = keypoints[2]
    if nose[2]<0.5 or l_eye[2]<0.5 or r_eye[2]<0.5:
        return False
    eye_cx   = (l_eye[0]+r_eye[0])/2.0
    eye_dist = abs(l_eye[0]-r_eye[0])+1e-6
    h_ratio  = abs(nose[0]-eye_cx)/eye_dist
    v_ratio  = (nose[1]-(l_eye[1]+r_eye[1])/2.0)/eye_dist
    return h_ratio < 0.15 and v_ratio > 0.40

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("="*60)
    print("STAGE 2 — Pose-Based Engagement Detection")
    print("CUDA: " + str(torch.cuda.is_available()))
    print("="*60)

    if not os.path.exists(STAGE1_METADATA):
        raise FileNotFoundError("Run stage1 first. Missing: " + STAGE1_METADATA)

    with open(STAGE1_METADATA,"r") as f:
        meta = json.load(f)

    if not meta:
        print("No persons in Stage 1. Exiting.")
        return

    print("Persons from Stage 1 : {}".format(len(meta)))
    print("Engagement threshold : {}s = {} frames".format(
        LOOK_THRESHOLD_SEC, FRAMES_REQUIRED))

    model, backend = load_model("yolov8n-pose")
    print("Backend: " + backend)

    results_out = {}
    t0 = time.time()
    n  = len(meta)

    for idx,(pid,pdata) in enumerate(meta.items()):
        pdir = pdata.get("person_dir","")

        def make_result(lf=0,tf=0):
            lts = lf/FPS_EQUIVALENT
            return {"looking_frames":lf,"total_frames":tf,
                    "looking_time_sec":round(lts,2),
                    "is_engaged": lf>=FRAMES_REQUIRED,
                    "roi_time_sec":pdata.get("roi_time_sec",0.0),
                    "entry_time_sec":pdata.get("entry_time_sec",0.0),
                    "person_dir":pdir}

        if not os.path.exists(pdir):
            print("  [WARN] Missing dir ID {}".format(pid))
            results_out[pid] = make_result(); continue

        crops = sorted([f for f in os.listdir(pdir)
                        if f.lower().endswith((".jpg",".png"))])
        if not crops:
            print("  [WARN] No crops ID {}".format(pid))
            results_out[pid] = make_result(); continue

        print("  [{}/{}] ID {} — {} crops".format(idx+1,n,pid,len(crops)))

        looking = 0
        total   = 0

        for cf in crops:
            img = cv2.imread(os.path.join(pdir,cf))
            if img is None or img.size==0: continue
            total += 1
            try:
                res = model(img, verbose=False, conf=POSE_CONF)
            except Exception:
                continue
            for r in res:
                if r.keypoints is None or r.keypoints.data.shape[0]==0: continue
                kpts = r.keypoints.data.cpu().numpy()[0]
                if kpts.shape[0]>=3 and is_looking(kpts):
                    looking += 1; break

        lts     = looking/FPS_EQUIVALENT
        engaged = looking >= FRAMES_REQUIRED
        results_out[pid] = {"looking_frames":looking,"total_frames":total,
                            "looking_time_sec":round(lts,2),
                            "is_engaged":engaged,
                            "roi_time_sec":pdata.get("roi_time_sec",0.0),
                            "entry_time_sec":pdata.get("entry_time_sec",0.0),
                            "person_dir":pdir}
        status = "ENGAGED" if engaged else "NOT ENGAGED"
        print("    {}/{} looking ({:.1f}s) → {}".format(looking,total,lts,status))

    with open(STAGE2_OUTPUT,"w") as f:
        json.dump(results_out,f,indent=2)

    engaged_n = sum(1 for v in results_out.values() if v["is_engaged"])
    elapsed   = time.time()-t0
    print("\n✅ Stage 2 Complete! ({:.1f}s)".format(elapsed))
    print("   Backend         : " + backend)
    print("   Engaged persons : {}/{}".format(engaged_n,len(results_out)))
    print("   Saved to        : " + STAGE2_OUTPUT)

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("   GPU memory released")

if __name__ == "__main__":
    run()
