"""Very small intent router to decide whether to call tools or the LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .actions import ActionExecutor
from .memory import MemoryManager


@dataclass(slots=True)
class Intent:
    name: str
    payload: dict


class IntentRouter:
    """Route common utterances to local tools instead of the LLM."""

    def __init__(self, memory: MemoryManager, actions: ActionExecutor) -> None:
        self._memory = memory
        self._actions = actions

    def route(self, text: str) -> Optional[str]:
        """Return a tool-driven response or ``None`` to fall back to the LLM."""

        intent = self._classify(text)
        if not intent:
            return None
        if intent.name == "remember_fact":
            fact: str = intent.payload["fact"]
            self._memory.remember_fact(fact)
            return "Got it! I'll remember that." \
                if fact.endswith(".") else f"Got it! I'll remember that {fact}."
        if intent.name == "recall_fact":
            query: str = intent.payload["query"]
            matches = self._memory.search_facts(query)
            if not matches:
                return "I couldn't find anything about that yet."
            formatted = "; ".join(matches)
            return f"Here's what I have: {formatted}"
        if intent.name == "add_todo":
            description: str = intent.payload["description"]
            due_at: Optional[datetime] = intent.payload.get("due_at")
            todo_id = self._memory.add_todo(description, due_at)
            if due_at:
                return f"Added todo {todo_id} for {due_at:%Y-%m-%d %H:%M}."
            return f"Added todo {todo_id}."
        if intent.name == "list_todos":
            todos = self._memory.list_todos()
            if not todos:
                return "Your todo list is empty."
            parts = []
            for todo in todos:
                due = f" (due {todo.due_at:%b %d %H:%M})" if todo.due_at else ""
                parts.append(f"#{todo.id}: {todo.description}{due}")
            return "Here are your todos: " + "; ".join(parts)
        if intent.name == "schedule_reminder":
            reminder = self._actions.schedule_reminder(**intent.payload)
            return f"Reminder set for {reminder.when:%Y-%m-%d %H:%M}."
        if intent.name == "checklist_add":
            checklist = self._actions.add_checklist_item(intent.payload["name"], intent.payload["item"])
            return f"Added to {checklist.name} checklist."
        if intent.name == "checklist_summary":
            checklist = self._actions.checklist_summary(intent.payload["name"])
            if not checklist:
                return "I don't have that checklist yet."
            return f"{checklist.name} checklist: " + "; ".join(checklist.items)
        return None

    def _classify(self, text: str) -> Optional[Intent]:
        lowered = text.lower().strip()
        if not lowered:
            return None
        if lowered.startswith("remember"):
            fact = text.partition("remember")[2].strip()
            if fact:
                return Intent(name="remember_fact", payload={"fact": fact})
        if lowered.startswith("what do you remember about"):
            query = text.partition("about")[2].strip()
            if query:
                return Intent(name="recall_fact", payload={"query": query})
        if lowered.startswith("add todo"):
            description = text.partition(":")[2].strip() or text.partition("add todo")[2].strip()
            if description:
                due_at = self._extract_datetime(text)
                return Intent(name="add_todo", payload={"description": description, "due_at": due_at})
        if lowered.startswith("list todos"):
            return Intent(name="list_todos", payload={})
        if lowered.startswith("remind me"):
            match = re.search(r"remind me (?:to )?(?P<message>.+?) (?:at|on|in) (?P<when>.+)", text, flags=re.IGNORECASE)
            if match:
                message = match.group("message").strip()
                when_text = match.group("when").strip()
                when = self._parse_datetime(when_text)
                if when:
                    return Intent(name="schedule_reminder", payload={"message": message, "when": when})
        if "checklist" in lowered and "add" in lowered:
            name, _, item = text.partition("checklist")
            item = item.replace("add", "", 1).strip(": ")
            if name and item:
                return Intent(name="checklist_add", payload={"name": name.strip(), "item": item})
        if lowered.endswith("checklist"):
            name = text.replace("checklist", "").strip()
            if name:
                return Intent(name="checklist_summary", payload={"name": name})
        return None

    def _extract_datetime(self, text: str) -> Optional[datetime]:
        when_match = re.search(r"(at|by|before|on) (.+)$", text, flags=re.IGNORECASE)
        if not when_match:
            return None
        return self._parse_datetime(when_match.group(2))

    def _parse_datetime(self, phrase: str) -> Optional[datetime]:
        from dateparser import parse

        parsed = parse(phrase, settings={"PREFER_DATES_FROM": "future"})
        return parsed
