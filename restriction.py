import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO
import threading
from dotenv import load_dotenv
from playsound import playsound
from db_logger import init_db, log_event_to_db
# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
GEOFENCE_ZONE = None
drawing = False
zone_start = (0, 0)
zone_temp = None

CONFIDENCE_THRESHOLD = 0.5
LOG_FILE = "geofence_log.csv"
MODEL_PATH = "yolov8n.pt"              # auto-downloads if not present
PERSON_CLASS_ID = 0                    # COCO class 0 = person
SNAPSHOT_DIR = "intrusion_snapshots"
# Colors (BGR)
COLOR_ZONE_SAFE    = (0, 255, 0)       # Green — no intrusion
COLOR_ZONE_ALERT   = (0, 0, 255)       # Red   — person inside
COLOR_PERSON_SAFE  = (255, 200, 0)     # Blue  — person outside zone
COLOR_PERSON_ALERT = (0, 0, 255)       # Red   — person inside zone
COLOR_TEXT         = (255, 255, 255)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))  # default 0 if not set
init_db()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def init_log(path: str):
    """Create CSV log with header if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event", "person_bbox", "confidence"])


def log_event(path: str, event: str, bbox: tuple, conf: float):
    """Append an intrusion event to the CSV log."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         event, str(bbox), f"{conf:.2f}"])
def save_snapshot(frame, directory: str):
    """Save current frame as an image when intrusion is detected."""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = os.path.join(directory, f"intrusion_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    return filename

def boxes_overlap(boxA: tuple, boxB: tuple) -> bool:
    """Return True if two (x1,y1,x2,y2) boxes overlap."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def draw_geofence(frame, zone: tuple, intruder: bool):
    """Draw the geofence rectangle on the frame."""
    color = COLOR_ZONE_ALERT if intruder else COLOR_ZONE_SAFE
    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = "⚠ RESTRICTED ZONE — INTRUSION!" if intruder else "RESTRICTED ZONE"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def draw_person(frame, bbox: tuple, conf: float, inside: bool):
    """Draw the person bounding box."""
    x1, y1, x2, y2 = bbox
    color = COLOR_PERSON_ALERT if inside else COLOR_PERSON_SAFE
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f"Person {'⚠' if inside else ''} {conf:.0%}"
    cv2.putText(frame, tag, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_overlay(frame, intruder_count: int, total_persons: int):
    """Draw HUD info in the top-left corner."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"Persons detected : {total_persons}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    cv2.putText(frame, f"In restricted zone: {intruder_count}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLOR_ZONE_ALERT if intruder_count else COLOR_TEXT, 1)

    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 90, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)


def flash_alert(frame):
    """Overlay a semi-transparent red tint when intrusion is detected."""
    alert = frame.copy()
    cv2.rectangle(alert, (0, 0), (frame.shape[1], frame.shape[0]),
                  (0, 0, 200), -1)
    cv2.addWeighted(alert, 0.15, frame, 0.85, 0, frame)
def play_alert_sound():
    """Play beep sound non-blocking, works on Windows/Linux/Mac."""
    def _play():
        try:
            playsound("beep.wav")
        except Exception as e:
            print(f"[WARN] Sound failed: {e}")
    threading.Thread(target=_play, daemon=True).start()
    
def select_zone_callback(event, x, y, flags, param):
    """Mouse callback to let user draw the geofence zone with click+drag."""
    global drawing, zone_start, zone_temp, GEOFENCE_ZONE

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        zone_start = (x, y)
        zone_temp = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        zone_temp = (zone_start[0], zone_start[1], x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = zone_start
        x2, y2 = x, y
        GEOFENCE_ZONE = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        zone_temp = None
def setup_geofence_zone(cap):
    """Let the user draw the geofence zone on a live preview before starting detection."""
    global GEOFENCE_ZONE, zone_temp

    cv2.namedWindow("Draw Geofence Zone - Click & Drag, press ENTER when done")
    cv2.setMouseCallback("Draw Geofence Zone - Click & Drag, press ENTER when done",
                          select_zone_callback)

    print("[INFO] Draw your geofence zone: click & drag, then press ENTER to confirm.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        if zone_temp:
            x1, y1, x2, y2 = zone_temp
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        elif GEOFENCE_ZONE:
            x1, y1, x2, y2 = GEOFENCE_ZONE
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(display, "Draw zone, then press ENTER", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Draw Geofence Zone - Click & Drag, press ENTER when done", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and GEOFENCE_ZONE:   # ENTER key, zone must be set
            break
        elif key == ord("q"):
            raise SystemExit("Zone setup cancelled by user.")

    cv2.destroyWindow("Draw Geofence Zone - Click & Drag, press ENTER when done")
    print(f"[INFO] Geofence zone set to: {GEOFENCE_ZONE}")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("[INFO] Loading YOLOv8 model …")
    model = YOLO(MODEL_PATH)

    print("[INFO] Opening webcam …")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check your device index.")
    setup_geofence_zone(cap)
    init_log(LOG_FILE)
    print(f"[INFO] Logging events to: {LOG_FILE}")
    print("[INFO] Press  Q  to quit.")

    prev_intruder = False   # track state change to avoid duplicate logs

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — skipping.")
            continue

        # ── Run YOLOv8 inference (only person class) ──────────────────────
        results = model(frame, classes=[PERSON_CLASS_ID],
                        conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        intruder_count = 0
        total_persons  = 0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            person_bbox = (x1, y1, x2, y2)

            total_persons += 1
            inside = boxes_overlap(person_bbox, GEOFENCE_ZONE)

            if inside:
                intruder_count += 1

            draw_person(frame, person_bbox, conf, inside)

            # Log only on state change (entered zone)
            if inside and not prev_intruder:
                snapshot_path = save_snapshot(frame, SNAPSHOT_DIR)
                log_event(LOG_FILE, "INTRUSION_DETECTED", person_bbox, conf)
                log_event_to_db("INTRUSION_DETECTED", bbox=person_bbox, confidence=conf, snapshot_path=snapshot_path)
                play_alert_sound()
                
                print(f"[ALERT] {datetime.now():%H:%M:%S} — Person entered restricted zone!")

        # ── Draw geofence & HUD ───────────────────────────────────────────
        intruding = intruder_count > 0
        draw_geofence(frame, GEOFENCE_ZONE, intruding)
        draw_overlay(frame, intruder_count, total_persons)

        if intruding:
            flash_alert(frame)

        # Log when person leaves zone
        if prev_intruder and not intruding:
            log_event(LOG_FILE, "ZONE_CLEARED", (0, 0, 0, 0), 0.0)
            log_event_to_db("ZONE_CLEARED")
            print(f"[INFO]  {datetime.now():%H:%M:%S} — Restricted zone cleared.")

        prev_intruder = intruding

        cv2.imshow("Geofence Restriction — YOLOv8", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()