import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase): ...


class Vertical(enum.StrEnum):
    education = "education"
    realestate = "realestate"


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    vertical: Mapped[Vertical] = mapped_column(Enum(Vertical, name="vertical"))
    default_lang: Mapped[str] = mapped_column(String(8), default="ar")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
