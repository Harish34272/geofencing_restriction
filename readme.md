# geofencing-restriction

Real-time geofencing restriction system using YOLOv8 person detection on a live webcam feed.
When a person enters a user-drawn restricted zone, the system triggers an audio alert, saves a snapshot, and logs the event to PostgreSQL.

---

## Features

- Mouse click-and-drag to draw the restricted zone on the live feed
- YOLOv8 person detection (COCO class 0)
- Audio alert via `playsound` + `beep.wav`
- Snapshot saved to `intrusion_snapshots/` on intrusion
- Event logging to PostgreSQL and optional CSV backup
- Web app: live camera view and analytics dashboard (port 5003)

---

## Repository

```bash
git clone https://github.com/Harish34272/geofencing_restriction.git
cd geofencing_restriction
```

---

## Environment Setup

```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt
```

**Ubuntu only** — install gstreamer for audio support:
```bash
sudo apt install python3-gst-1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

---

## Dependencies

| Package | Version |
| --- | --- |
| alembic | 1.19.1 |
| opencv-python | 5.0.0.93 |
| playsound | 1.2.2 |
| psycopg2-binary | 2.9.12 |
| python-dotenv | 1.2.3 |
| SQLAlchemy | 2.0.52 |
| ultralytics | 8.4.154 |
| ultralytics-thop | 2.1.6 |

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=geofence_db
DB_USER=postgres
DB_PASSWORD=your_password_here

CAMERA_INDEX=0
CONFIDENCE_THRESHOLD=0.5
SNAPSHOT_DIR=intrusion_snapshots
ENABLE_CSV_LOG=true
```

---

## Generate beep.wav

Run once to create the alert sound file:

```bash
python generate_beep.py
```

---

## PostgreSQL Setup

### Windows

Start PostgreSQL in an admin terminal:

```powershell
net start postgresql-x64-18
```

If `psql` is not in your PATH (non-standard install location), use the full path:

```powershell
& "D:\language-programs\PostgreSQL\18\bin\psql.exe" -h localhost -p 5432 -U postgres -d postgres
```

To find `psql.exe` on your machine:

```powershell
where.exe /R "D:\language-programs" psql.exe
```

### Create the database

```sql
CREATE DATABASE geofence_db;
\q
```

### Ubuntu / Linux

```bash
sudo service postgresql start
psql -U postgres -c "CREATE DATABASE geofence_db;"
```

---

## Database Migration

Run Alembic migrations to create the tables:

```bash
alembic upgrade head
```

> The `init_db()` call in `restriction.py` also creates tables automatically on first run if Alembic is not yet wired.

---

## Running

Activate your environment, then:

```bash
# Main detection script
conda activate <env_name>
python restriction.py

# Web app (live camera + dashboard)
python app.py
```

Web app pages:
- `http://<ip>:5003/live_camera` — live webcam feed with geofence overlay
- `http://<ip>:5003/dashboard` — intrusion event log and snapshots

---

## .gitignore

```
.env
__pycache__/*
zone_state.json
geofencing_log.csv
intrusion_snapshots/
```