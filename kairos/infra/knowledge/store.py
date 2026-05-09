"""Knowledge schema framework — user-defined typed schemas with structured storage."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Type

from kairos.infra.knowledge.schema import KnowledgeSchema


class KnowledgeStore:
    """
    Structured knowledge store with typed schema support.

    Usage:
        class FaultDiagnosis(KnowledgeSchema):
            signal_name: str
            root_cause: str
            solution: str

        store = KnowledgeStore(schema=FaultDiagnosis)
        store.insert(FaultDiagnosis(id="F-001", signal_name="engine_temp", ...))
        results = store.query({"root_cause": "controller_overheat"})
    """

    def __init__(self, schema: Type[KnowledgeSchema]):
        self.schema = schema
        self._items: dict[str, KnowledgeSchema] = {}

    def insert(self, item: KnowledgeSchema) -> None:
        """Insert or update a knowledge item."""
        item.updated_at = datetime.now()
        self._items[item.id] = item

    def get(self, item_id: str) -> KnowledgeSchema | None:
        """Get a single item by ID."""
        return self._items.get(item_id)

    def query(self, filters: dict[str, Any] | None = None) -> list[KnowledgeSchema]:
        """Query items by field-value filters. Returns all if no filters given."""
        if not filters:
            return list(self._items.values())
        results = []
        for item in self._items.values():
            match = True
            for key, value in filters.items():
                if not hasattr(item, key) or getattr(item, key) != value:
                    match = False
                    break
            if match:
                results.append(item)
        return results

    def search(self, text: str) -> list[KnowledgeSchema]:
        """Full-text search across all string fields."""
        results = []
        text_lower = text.lower()
        for item in self._items.values():
            item_str = str(item.to_dict()).lower()
            if text_lower in item_str:
                results.append(item)
        return results

    def delete(self, item_id: str) -> bool:
        """Delete an item. Returns True if found and deleted."""
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    def count(self) -> int:
        return len(self._items)
