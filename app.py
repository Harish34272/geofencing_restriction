"""
app.py — Web layer for the geofencing restriction project.

Routes:
    /live_camera   Live MJPEG feed with zone + bounding boxes, zone drawn in-browser
    /video_feed    Raw MJPEG stream (multipart/x-mixed-replace)
    /frame_info    JSON — native camera resolution, used to scale browser clicks
    /set_zone      POST — set geofence zone from browser rectangle
    /clear_zone    POST — clear the current zone
    /status        JSON — current alert status (polled by the live page)
    /dashboard     Event log, stats, snapshot thumbnails from PostgreSQL
    /snapshots/<f> Serves a saved .jpg snapshot from intrusion_snapshots/

Run:
    python app.py
Then open:
    http://<ip>:5003/live_camera
    http://<ip>:5003/dashboard
"""

import os
import csv
import io
import json
import time
import threading
from datetime import datetime
from collections import Counter
from math import ceil

import cv2
from flask import (
    Flask, Response, render_template, request,
    jsonify, redirect, url_for, send_from_directory
)
from dotenv import load_dotenv
from ultralytics import YOLO
from playsound import playsound

from db_logger import init_db, log_event_to_db, get_recent_events, get_filtered_events

# ─────────────────────────────────────────
# CONFIG (mirrors restriction.py)
# ─────────────────────────────────────────
load_dotenv()

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
DEFAULT_CONFIDENCE = 0.5     # starting point; adjustable live from /live_camera
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 0.95
MODEL_PATH = "yolov8n.pt"
PERSON_CLASS_ID = 0
SNAPSHOT_DIR = "intrusion_snapshots"
LOG_FILE = "geofence_log.csv"
ENABLE_CSV_LOG = os.getenv("ENABLE_CSV_LOG", "true").lower() == "true"
ZONE_STATE_FILE = "zone_state.json"
JPEG_QUALITY = 80
STREAM_FPS_CAP = 20          # caps how often we re-encode/send a frame

COLOR_ZONE_SAFE    = (0, 255, 0)
COLOR_ZONE_ALERT   = (0, 0, 255)
COLOR_PERSON_SAFE  = (255, 200, 0)
COLOR_PERSON_ALERT = (0, 0, 255)
COLOR_TEXT         = (255, 255, 255)

app = Flask(__name__)

# ─────────────────────────────────────────
# SHARED STATE (guarded by state_lock)
# ─────────────────────────────────────────
state_lock = threading.Lock()
latest_jpeg = None                # bytes, most recent encoded frame
frame_size = {"width": None, "height": None}
geofence_zone = None              # (x1, y1, x2, y2) in native frame coordinates
detection_confidence = DEFAULT_CONFIDENCE   # read fresh each frame by camera_worker
status = {
    "intruding": False,
    "persons_detected": 0,
    "intruders": 0,
    "last_event": None,
    "last_event_time": None,
    "zone_set": False,
    "confidence": DEFAULT_CONFIDENCE,
}


# ─────────────────────────────────────────
# ZONE PERSISTENCE — preload last session's zone on startup
# ─────────────────────────────────────────
def load_saved_zone():
    if os.path.exists(ZONE_STATE_FILE):
        try:
            with open(ZONE_STATE_FILE, "r") as f:
                data = json.load(f)
                z = data.get("zone")
                if z and len(z) == 4:
                    return tuple(z)
        except Exception as e:
            print(f"[WARN] Could not read {ZONE_STATE_FILE}: {e}")
    return None


def save_zone_to_disk(zone):
    try:
        with open(ZONE_STATE_FILE, "w") as f:
            json.dump({"zone": list(zone), "saved_at": datetime.now().isoformat()}, f)
    except Exception as e:
        print(f"[WARN] Could not save zone: {e}")


# ─────────────────────────────────────────
# HELPERS (same logic as restriction.py)
# ─────────────────────────────────────────
def init_log(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event", "person_bbox", "confidence"])


def log_event_csv(path: str, event: str, bbox: tuple, conf: float):
    if not ENABLE_CSV_LOG:
        return
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          event, str(bbox), f"{conf:.2f}"])


def save_snapshot(frame, directory: str):
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = os.path.join(directory, f"intrusion_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    return filename


def boxes_overlap(boxA: tuple, boxB: tuple) -> bool:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def draw_geofence(frame, zone, intruder: bool):
    color = COLOR_ZONE_ALERT if intruder else COLOR_ZONE_SAFE
    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = "INTRUSION - RESTRICTED ZONE" if intruder else "RESTRICTED ZONE"
    cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_person(frame, bbox, conf, inside):
    x1, y1, x2, y2 = bbox
    color = COLOR_PERSON_ALERT if inside else COLOR_PERSON_SAFE
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f"Person {conf:.0%}"
    cv2.putText(frame, tag, (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_overlay(frame, intruder_count, total_persons):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"Persons: {total_persons}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    cv2.putText(frame, f"In zone: {intruder_count}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLOR_ZONE_ALERT if intruder_count else COLOR_TEXT, 1)
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 85, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)


def flash_alert(frame):
    alert = frame.copy()
    cv2.rectangle(alert, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
    cv2.addWeighted(alert, 0.15, frame, 0.85, 0, frame)


def play_alert_sound():
    def _play():
        try:
            playsound("beep.wav")
        except Exception as e:
            print(f"[WARN] Sound failed: {e}")
    threading.Thread(target=_play, daemon=True).start()


def draw_no_zone_banner(frame):
    cv2.putText(frame, "No geofence zone set — draw one below",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)


# ─────────────────────────────────────────
# BACKGROUND CAMERA / DETECTION WORKER
# ─────────────────────────────────────────
def camera_worker():
    global latest_jpeg, geofence_zone

    print("[INFO] Loading YOLOv8 model …")
    model = YOLO(MODEL_PATH)

    print("[INFO] Opening webcam …")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check CAMERA_INDEX in .env.")
        return

    init_log(LOG_FILE)

    with state_lock:
        geofence_zone = load_saved_zone()
        status["zone_set"] = geofence_zone is not None

    prev_intruder = False
    min_frame_interval = 1.0 / STREAM_FPS_CAP
    last_sent = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — retrying.")
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        with state_lock:
            frame_size["width"] = w
            frame_size["height"] = h
            zone = geofence_zone
            conf_threshold = detection_confidence

        results = model(frame, classes=[PERSON_CLASS_ID],
                         conf=conf_threshold, verbose=False)[0]

        intruder_count = 0
        total_persons = 0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            person_bbox = (x1, y1, x2, y2)
            total_persons += 1

            inside = zone is not None and boxes_overlap(person_bbox, zone)
            if inside:
                intruder_count += 1

            draw_person(frame, person_bbox, conf, inside)

            if inside and not prev_intruder:
                snapshot_path = save_snapshot(frame, SNAPSHOT_DIR)
                log_event_csv(LOG_FILE, "INTRUSION_DETECTED", person_bbox, conf)
                log_event_to_db("INTRUSION_DETECTED", bbox=person_bbox,
                                 confidence=conf, snapshot_path=snapshot_path)
                play_alert_sound()
                print(f"[ALERT] {datetime.now():%H:%M:%S} — Person entered restricted zone!")
                with state_lock:
                    status["last_event"] = "INTRUSION_DETECTED"
                    status["last_event_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if zone is not None:
            draw_geofence(frame, zone, intruder_count > 0)
        else:
            draw_no_zone_banner(frame)

        draw_overlay(frame, intruder_count, total_persons)

        intruding = intruder_count > 0
        if intruding:
            flash_alert(frame)

        if prev_intruder and not intruding:
            log_event_csv(LOG_FILE, "ZONE_CLEARED", (0, 0, 0, 0), 0.0)
            log_event_to_db("ZONE_CLEARED")
            print(f"[INFO] {datetime.now():%H:%M:%S} — Restricted zone cleared.")
            with state_lock:
                status["last_event"] = "ZONE_CLEARED"
                status["last_event_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prev_intruder = intruding

        with state_lock:
            status["intruding"] = intruding
            status["persons_detected"] = total_persons
            status["intruders"] = intruder_count
            status["zone_set"] = zone is not None
            status["confidence"] = conf_threshold

        # Throttle encoding/sending to STREAM_FPS_CAP
        now = time.time()
        if now - last_sent >= min_frame_interval:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with state_lock:
                    latest_jpeg = buf.tobytes()
            last_sent = now


_worker_started = False
_worker_lock = threading.Lock()


def ensure_worker_started():
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=camera_worker, daemon=True).start()
            _worker_started = True


# ─────────────────────────────────────────
# ROUTES — LIVE CAMERA
# ─────────────────────────────────────────
@app.route("/")
def root():
    return redirect(url_for("live_camera"))


@app.route("/live_camera")
def live_camera():
    ensure_worker_started()
    return render_template("live_camera.html")


def gen_frames():
    boundary = b"--frame\r\n"
    while True:
        with state_lock:
            frame_bytes = latest_jpeg
        if frame_bytes is not None:
            yield (boundary + b"Content-Type: image/jpeg\r\n\r\n" +
                   frame_bytes + b"\r\n")
        time.sleep(1.0 / STREAM_FPS_CAP)


@app.route("/video_feed")
def video_feed():
    ensure_worker_started()
    return Response(gen_frames(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/frame_info")
def frame_info():
    with state_lock:
        w, h = frame_size["width"], frame_size["height"]
    if w is None:
        return jsonify({"ready": False}), 503
    return jsonify({"ready": True, "width": w, "height": h})


@app.route("/set_zone", methods=["POST"])
def set_zone():
    global geofence_zone
    data = request.get_json(force=True)
    try:
        x1, y1, x2, y2 = data["x1"], data["y1"], data["x2"], data["y2"]
        disp_w, disp_h = data["display_width"], data["display_height"]
    except (KeyError, TypeError):
        return jsonify({"ok": False, "error": "Missing coordinates"}), 400

    with state_lock:
        fw, fh = frame_size["width"], frame_size["height"]

    if fw is None:
        return jsonify({"ok": False, "error": "Camera not ready yet"}), 503

    # Scale from the browser's displayed <img> size to native frame resolution
    scale_x = fw / disp_w
    scale_y = fh / disp_h
    zone = (
        int(min(x1, x2) * scale_x), int(min(y1, y2) * scale_y),
        int(max(x1, x2) * scale_x), int(max(y1, y2) * scale_y),
    )

    if zone[2] - zone[0] < 10 or zone[3] - zone[1] < 10:
        return jsonify({"ok": False, "error": "Zone too small"}), 400

    with state_lock:
        geofence_zone = zone
    save_zone_to_disk(zone)
    return jsonify({"ok": True, "zone": zone})


@app.route("/set_confidence", methods=["POST"])
def set_confidence():
    global detection_confidence
    data = request.get_json(force=True)
    try:
        value = float(data["confidence"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "Missing or invalid 'confidence'"}), 400

    if not (MIN_CONFIDENCE <= value <= MAX_CONFIDENCE):
        return jsonify({"ok": False,
                         "error": f"confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}"}), 400

    with state_lock:
        detection_confidence = value
    return jsonify({"ok": True, "confidence": value})


@app.route("/clear_zone", methods=["POST"])
def clear_zone():
    global geofence_zone
    with state_lock:
        geofence_zone = None
    if os.path.exists(ZONE_STATE_FILE):
        os.remove(ZONE_STATE_FILE)
    return jsonify({"ok": True})


@app.route("/status")
def get_status():
    with state_lock:
        return jsonify(dict(status))


# ─────────────────────────────────────────
# ROUTES — DASHBOARD
# ─────────────────────────────────────────
DASHBOARD_FETCH_CAP = 200   # how many filtered events we pull before paginating
PER_PAGE = 10


def _read_dashboard_filters():
    """Parse & validate filter/pagination query params shared by /dashboard and /dashboard/export."""
    event_type = request.args.get("event_type") or None
    if event_type not in (None, "INTRUSION_DETECTED", "ZONE_CLEARED"):
        event_type = None

    min_conf_raw = request.args.get("min_conf", "")
    try:
        min_conf = float(min_conf_raw) if min_conf_raw != "" else None
    except ValueError:
        min_conf = None

    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    return event_type, min_conf, page


@app.route("/dashboard")
def dashboard():
    event_type, min_conf, page = _read_dashboard_filters()

    events = get_filtered_events(limit=DASHBOARD_FETCH_CAP,
                                  event_type=event_type, min_confidence=min_conf)

    total_intrusions = sum(1 for e in events if e.event_type == "INTRUSION_DETECTED")
    last_event_time = events[0].timestamp.strftime("%Y-%m-%d %H:%M:%S") if events else "—"

    total_pages = max(1, ceil(len(events) / PER_PAGE))
    page = min(page, total_pages)
    page_events = events[(page - 1) * PER_PAGE: page * PER_PAGE]

    rows = []
    for e in page_events:
        rows.append({
            "id": e.id,
            "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": e.event_type,
            "confidence": f"{e.confidence:.0%}" if e.confidence is not None else "—",
            "snapshot_filename": (os.path.basename(e.snapshot_path)
                                   if e.snapshot_path else None),
        })

    # Intrusions by hour-of-day, across the filtered set (not just this page)
    hour_counts = Counter(
        e.timestamp.hour for e in events if e.event_type == "INTRUSION_DETECTED"
    )
    max_hour_count = max(hour_counts.values(), default=0)
    hourly = [{
        "hour": f"{h:02d}",
        "count": hour_counts.get(h, 0),
        "pct": round((hour_counts.get(h, 0) / max_hour_count) * 100) if max_hour_count else 0,
    } for h in range(24)]

    return render_template(
        "dashboard.html",
        rows=rows,
        total_intrusions=total_intrusions,
        total_events=len(events),
        last_event_time=last_event_time,
        page=page,
        total_pages=total_pages,
        event_type=event_type or "",
        min_conf=min_conf,
        hourly=hourly,
    )


@app.route("/dashboard/export")
def dashboard_export():
    event_type, min_conf, _ = _read_dashboard_filters()
    events = get_filtered_events(limit=DASHBOARD_FETCH_CAP,
                                  event_type=event_type, min_confidence=min_conf)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "event_type", "bbox", "confidence", "snapshot_path"])
    for e in events:
        writer.writerow([
            e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            e.event_type,
            e.bbox or "",
            f"{e.confidence:.4f}" if e.confidence is not None else "",
            e.snapshot_path or "",
        ])

    filename = f"geofence_events_{datetime.now():%Y%m%d_%H%M%S}.csv"
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/snapshots/<path:filename>")
def snapshots(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    ensure_worker_started()
    app.run(host="0.0.0.0", port=5003, debug=False, threaded=True)