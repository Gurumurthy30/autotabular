import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session, select

from app.db.models import Event
from app.db.session import engine


class EventManager:
    """Manages event persistence to SQLite and in-memory fan-out to SSE subscribers."""

    def __init__(self):
        # Mapping: (project_id, run_id) -> list of asyncio.Queue
        self._subscribers: dict[tuple[str, str], list[asyncio.Queue]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def emit_event(
        self,
        project_id: str,
        run_id: str,
        event_type: str,
        stage: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persists the event to SQLite and broadcasts to all active SSE queues."""
        event_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        data_payload = data or {}

        event_record = Event(
            id=event_id,
            project_id=project_id,
            run_id=run_id,
            event_type=event_type,
            stage=stage,
            message=message,
            data_json=json.dumps(data_payload),
            timestamp=now,
        )

        with Session(engine) as session:
            session.add(event_record)
            session.commit()

        event_dict = {
            "id": event_id,
            "project_id": project_id,
            "run_id": run_id,
            "event_type": event_type,
            "stage": stage,
            "message": message,
            "data": data_payload,
            "timestamp": now.isoformat(),
        }

        # Fan-out to memory queues thread-safely
        key = (project_id, run_id)
        queues = self._subscribers.get(key, [])
        for q in list(queues):
            try:
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(q.put_nowait, event_dict)
                else:
                    q.put_nowait(event_dict)
            except Exception:
                pass

        return event_dict

    def get_past_events(self, project_id: str, run_id: str) -> list[dict[str, Any]]:
        """Retrieves stored historical events for a run in chronological order."""
        with Session(engine) as session:
            statement = (
                select(Event)
                .where(Event.project_id == project_id, Event.run_id == run_id)
                .order_by(Event.timestamp.asc())
            )
            rows = session.exec(statement).all()
            return [
                {
                    "id": r.id,
                    "project_id": r.project_id,
                    "run_id": r.run_id,
                    "event_type": r.event_type,
                    "stage": r.stage,
                    "message": r.message,
                    "data": json.loads(r.data_json or "{}"),
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in rows
            ]

    def subscribe(self, project_id: str, run_id: str) -> asyncio.Queue:
        """Registers a new SSE client subscriber queue."""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        q = asyncio.Queue()
        key = (project_id, run_id)
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(q)
        return q

    def unsubscribe(self, project_id: str, run_id: str, q: asyncio.Queue) -> None:
        """Removes an SSE client subscriber queue."""
        key = (project_id, run_id)
        if key in self._subscribers and q in self._subscribers[key]:
            self._subscribers[key].remove(q)
            if not self._subscribers[key]:
                del self._subscribers[key]


# Global singleton instance
event_manager = EventManager()
