from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class IntrusionEvent(Base):
    """
    Represents a single geofence event (intrusion or zone-cleared)
    detected by the YOLOv8 geofencing script.
    """
    __tablename__ = "intrusion_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    event_type = Column(String(50), nullable=False)        # e.g. INTRUSION_DETECTED, ZONE_CLEARED
    bbox = Column(String(100), nullable=True)              # "(x1, y1, x2, y2)"
    confidence = Column(Float, nullable=True)
    snapshot_path = Column(String(255), nullable=True)     # path to saved .jpg, not the image itself

    def __repr__(self):
        return (f"<IntrusionEvent(id={self.id}, event_type='{self.event_type}', "
                f"timestamp={self.timestamp}, snapshot_path='{self.snapshot_path}')>")