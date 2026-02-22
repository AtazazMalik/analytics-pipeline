#!/usr/bin/env python3
# ============================================================================
# STAGE 3 — DeepFace Demographic Analysis (Engaged Persons Only)
# Matches updated notebook: Gender + Age using Counter voting
# CPU inference | Jetson Nano | Python 3.6
# ============================================================================
import cv2, os, json, csv, time
import numpy as np
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from deepface import DeepFace

# ── Config ────────────────────────────────────────────────────────────────────
STAGE2_INPUT     = "/app/stage2_engagement.json"
FINAL_OUTPUT_CSV = "/app/outputs/final_analytics.csv"
MIN_CROP_SIZE    = 40
MAX_CROPS_PER_ID = 60       # cap for Jetson speed
DETECTOR_BACKEND = "opencv" # fastest on Jetson Nano

# ── Face analysis (no cache — matches new notebook logic) ─────────────────────
def analyze_face(img):
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
        return None
    try:
        result = DeepFace.analyze(
            img,
            actions=["age","gender"],
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
            silent=True
        )
        if isinstance(result, list):
            result = result[0]
        return {
            "gender": result.get("dominant_gender","N/A"),
            "age":    int(result.get("age",0))
        }
    except Exception:
        return None

# ── Helpers ───────────────────────────────────────────────────────────────────
def empty_row(pid, roi_t, look_t):
    return {"ID":pid,"ROI_Time":roi_t,"Looking_Time":look_t,
            "Gender":"N/A","Age":"N/A"}

def write_csv(rows):
    os.makedirs(os.path.dirname(FINAL_OUTPUT_CSV), exist_ok=True)
    with open(FINAL_OUTPUT_CSV,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID","ROI_Time","Looking_Time","Gender","Age"])
        w.writeheader(); w.writerows(rows)

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("="*60)
    print("STAGE 3 — DeepFace Demographic Analysis (CPU)")
    print("="*60)

    if not os.path.exists(STAGE2_INPUT):
        raise FileNotFoundError("Run stage2 first. Missing: " + STAGE2_INPUT)

    with open(STAGE2_INPUT,"r") as f:
        eng_data = json.load(f)

    engaged = {pid:d for pid,d in eng_data.items() if d.get("is_engaged",False)}
    print("Total: {} | Engaged: {}".format(len(eng_data),len(engaged)))

    if not engaged:
        print("No engaged persons. Writing empty CSV.")
        write_csv([]); return

    rows = []
    t0   = time.time()
    n    = len(engaged)

    for idx,(pid,data) in enumerate(engaged.items()):
        pdir = data.get("person_dir","")
        roi_t  = data.get("roi_time_sec",0.0)
        look_t = data.get("looking_time_sec",0.0)

        print("\n  [{}/{}] ID {} | ROI:{:.1f}s | Looking:{:.1f}s".format(
            idx+1,n,pid,roi_t,look_t))

        if not os.path.exists(pdir):
            print("    [WARN] Missing dir"); rows.append(empty_row(pid,roi_t,look_t)); continue

        crops = sorted([f for f in os.listdir(pdir)
                        if f.lower().endswith((".jpg",".png"))])
        if not crops:
            print("    [WARN] No crops"); rows.append(empty_row(pid,roi_t,look_t)); continue

        # Sample crops — cap for Jetson speed
        step   = max(1, len(crops)//MAX_CROPS_PER_ID)
        sample = crops[::step][:MAX_CROPS_PER_ID]
        print("    Crops: {} | Sampled: {}".format(len(crops),len(sample)))

        age_samples    = []
        gender_samples = []
        valid_frames   = 0

        for cf in sample:
            img = cv2.imread(os.path.join(pdir,cf))
            res = analyze_face(img)
            if res is None: continue
            valid_frames += 1
            age_samples.append(res["age"])
            gender_samples.append(res["gender"])

        if valid_frames == 0:
            final_age    = "N/A"
            final_gender = "N/A"
        else:
            final_age    = int(np.mean(age_samples))
            final_gender = Counter(gender_samples).most_common(1)[0][0]

        print("    → Gender: {} | Age: {} (valid: {}/{})".format(
            final_gender, final_age, valid_frames, len(sample)))

        rows.append({"ID":pid,"ROI_Time":roi_t,"Looking_Time":look_t,
                     "Gender":final_gender,"Age":final_age})

    write_csv(rows)
    elapsed = time.time()-t0

    print("\n✅ Stage 3 Complete! ({:.1f}s)".format(elapsed))
    print("   Results saved: " + FINAL_OUTPUT_CSV)
    print("\n📊 FINAL ANALYTICS SUMMARY:")
    print("{:<8} {:<12} {:<15} {:<10} {}".format(
        "ID","ROI_Time","Looking_Time","Gender","Age"))
    print("-"*55)
    for r in rows:
        print("{:<8} {:<12} {:<15} {:<10} {}".format(
            r["ID"],r["ROI_Time"],r["Looking_Time"],r["Gender"],str(r["Age"])))

if __name__ == "__main__":
    run()
