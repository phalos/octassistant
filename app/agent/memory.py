"""Long-term memory management using SQLite for structure and Chroma for recall."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(slots=True)
class Todo:
    """Simple todo representation."""

    id: int
    description: str
    due_at: Optional[datetime]
    completed: bool


class MemoryManager:
    """Persist structured and unstructured memory for the assistant."""

    def __init__(
        self,
        db_path: str | Path = "data/memory.db",
        vector_path: str | Path = "data/vectordb",
        collection_name: str = "memories",
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._db_path)
        self._connection.row_factory = sqlite3.Row
        self._bootstrap_schema()

        from chromadb import PersistentClient
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._vector_client = PersistentClient(path=str(Path(vector_path)))
        self._collection = self._vector_client.get_or_create_collection(
            name=collection_name,
            embedding_function=SentenceTransformerEmbeddingFunction(),
        )

    def close(self) -> None:
        """Close underlying resources."""

        self._connection.close()

    def _bootstrap_schema(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                tags TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                due_at TEXT,
                completed INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def remember_fact(self, text: str, tags: Optional[Iterable[str]] = None) -> int:
        """Store a fact in SQLite and index it in the vector database."""

        tag_blob = json.dumps(list(tags or []))
        created_at = datetime.utcnow().isoformat()
        cursor = self._connection.cursor()
        cursor.execute(
            "INSERT INTO facts (text, tags, created_at) VALUES (?, ?, ?)",
            (text, tag_blob, created_at),
        )
        fact_id = cursor.lastrowid
        self._connection.commit()
        document_id = f"fact:{fact_id}"
        self._collection.upsert(documents=[text], ids=[document_id], metadatas=[{"tags": tag_blob}])
        return int(fact_id)

    def search_facts(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve similar memories using the vector database."""

        if not query.strip():
            return []
        results = self._collection.query(query_texts=[query], n_results=n_results)
        documents: List[str] = results.get("documents", [[]])[0]
        return [doc for doc in documents if doc]

    def add_todo(self, description: str, due_at: Optional[datetime] = None) -> int:
        """Insert a todo item and return its identifier."""

        cursor = self._connection.cursor()
        cursor.execute(
            "INSERT INTO todos (description, due_at, created_at) VALUES (?, ?, ?)",
            (description, due_at.isoformat() if due_at else None, datetime.utcnow().isoformat()),
        )
        todo_id = cursor.lastrowid
        self._connection.commit()
        return int(todo_id)

    def list_todos(self, include_completed: bool = False) -> List[Todo]:
        """Return todos sorted by due date."""

        cursor = self._connection.cursor()
        query = "SELECT id, description, due_at, completed FROM todos"
        if not include_completed:
            query += " WHERE completed = 0"
        query += " ORDER BY COALESCE(due_at, '9999-12-31T23:59:59')"
        rows = cursor.execute(query).fetchall()
        todos: List[Todo] = []
        for row in rows:
            due = datetime.fromisoformat(row["due_at"]) if row["due_at"] else None
            todos.append(
                Todo(
                    id=row["id"],
                    description=row["description"],
                    due_at=due,
                    completed=bool(row["completed"]),
                )
            )
        return todos

    def complete_todo(self, todo_id: int) -> None:
        """Mark a todo item as completed."""

        cursor = self._connection.cursor()
        cursor.execute("UPDATE todos SET completed = 1 WHERE id = ?", (todo_id,))
        self._connection.commit()
