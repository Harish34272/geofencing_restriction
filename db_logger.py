import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, IntrusionEvent

# ─────────────────────────────────────────
# LOAD CONFIG FROM .env
# ─────────────────────────────────────────
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "geofence_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ─────────────────────────────────────────
# ENGINE + SESSION SETUP
# ─────────────────────────────────────────
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """
    Create tables if they don't exist yet.
    NOTE: Once you're using Alembic for migrations, you typically stop
    relying on this and let Alembic manage schema creation/changes instead.
    Safe to keep for quick local testing.
    """
    Base.metadata.create_all(bind=engine)


def log_event_to_db(event_type: str, bbox: tuple = None, confidence: float = None,
                     snapshot_path: str = None):
    """
    Insert a single geofence event (intrusion or zone-cleared) into PostgreSQL.

    Args:
        event_type: e.g. "INTRUSION_DETECTED" or "ZONE_CLEARED"
        bbox: (x1, y1, x2, y2) tuple, or None
        confidence: detection confidence score, or None
        snapshot_path: path to the saved .jpg snapshot, or None

    Returns:
        The inserted IntrusionEvent object (with its new id), or None on failure.
    """
    session = SessionLocal()
    try:
        event = IntrusionEvent(
            event_type=event_type,
            bbox=str(bbox) if bbox else None,
            confidence=float(confidence) if confidence is not None else None,
            snapshot_path=snapshot_path
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        return event
    except Exception as e:
        session.rollback()
        print(f"[DB ERROR] Failed to log event: {e}")
        return None
    finally:
        session.close()


def get_recent_events(limit: int = 20):
    """Fetch the most recent N events, newest first. Useful for debugging/review."""
    session = SessionLocal()
    try:
        return (session.query(IntrusionEvent)
                .order_by(IntrusionEvent.timestamp.desc())
                .limit(limit)
                .all())
    finally:
        session.close()