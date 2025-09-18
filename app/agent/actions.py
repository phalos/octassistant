"""Intent implementations such as reminders, notes, and checklists."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger


@dataclass(slots=True)
class Reminder:
    """In-memory representation of a scheduled reminder."""

    job_id: str
    message: str
    when: datetime


@dataclass(slots=True)
class Checklist:
    """Checklist container for lightweight project tracking."""

    name: str
    items: List[str] = field(default_factory=list)


class ActionExecutor:
    """High-level façade responsible for executing assistant actions."""

    def __init__(
        self,
        notifier: Callable[[str], None],
        journal_path: str | Path = "data/journal/daily.jsonl",
    ) -> None:
        self._scheduler = BackgroundScheduler()
        self._notifier = notifier
        self._reminders: Dict[str, Reminder] = {}
        self._checklists: Dict[str, Checklist] = {}
        self._journal_path = Path(journal_path)
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown()

    def schedule_reminder(self, message: str, when: datetime) -> Reminder:
        """Schedule a reminder and return its representation."""

        trigger = DateTrigger(run_date=when)
        job = self._scheduler.add_job(self._emit_reminder, trigger=trigger, args=[message], kwargs={"job_id": None})
        job.modify(kwargs={"job_id": job.id})
        reminder = Reminder(job_id=job.id, message=message, when=when)
        self._reminders[job.id] = reminder
        return reminder

    def cancel_reminder(self, job_id: str) -> bool:
        """Cancel a scheduled reminder."""

        if job_id not in self._reminders:
            return False
        self._scheduler.remove_job(job_id)
        del self._reminders[job_id]
        return True

    def list_reminders(self) -> List[Reminder]:
        """Return all pending reminders."""

        return list(self._reminders.values())

    def append_journal(self, payload: Dict[str, object]) -> None:
        """Append a JSON line entry to the assistant journal."""

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            **payload,
        }
        with self._journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_or_create_checklist(self, name: str) -> Checklist:
        checklist = self._checklists.setdefault(name.lower(), Checklist(name=name))
        return checklist

    def add_checklist_item(self, name: str, item: str) -> Checklist:
        checklist = self.get_or_create_checklist(name)
        checklist.items.append(item)
        return checklist

    def checklist_summary(self, name: str) -> Optional[Checklist]:
        return self._checklists.get(name.lower())

    def _emit_reminder(self, message: str, job_id: str | None = None) -> None:
        self._notifier(message)
        identifier = job_id or next((key for key, value in self._reminders.items() if value.message == message), None)
        if identifier and identifier in self._reminders:
            del self._reminders[identifier]
