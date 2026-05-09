"""Knowledge schema — base class for user-defined typed knowledge domains."""

from __future__ import annotations

from datetime import datetime


class KnowledgeSchema:
    """Base class for all domain knowledge schemas.

    Subclass this to define your domain's structured knowledge:

        class FaultDiagnosis(KnowledgeSchema):
            signal_name: str
            root_cause: str
            solution: str

    Then store and query with KnowledgeStore:
        store = KnowledgeStore(FaultDiagnosis)
        store.insert(FaultDiagnosis(id="F-001", ...))
        results = store.query({"root_cause": "controller_overheat"})
    """

    id: str
    created_at: datetime
    updated_at: datetime

    def __init__(
        self,
        id: str,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ):
        self.id = id
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()

    def to_dict(self) -> dict:
        """Serialize to a plain dict for storage."""
        return {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeSchema:
        """Deserialize from a dict."""
        obj = cls.__new__(cls)
        for k, v in data.items():
            obj.__dict__[k] = v
        return obj

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if k != "id"
        )
        return f"{self.__class__.__name__}(id={self.id!r}, {fields})"
