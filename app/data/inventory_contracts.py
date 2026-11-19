"""Validated source contract for time-stamped inventory snapshots."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class InventorySnapshotInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    seller_id: str = Field(min_length=1, max_length=128)
    warehouse_id: str = Field(min_length=1, max_length=128)
    product_id: str = Field(min_length=1, max_length=128)
    sku: str | None = Field(default=None, max_length=256)
    snapshot_at: datetime
    snapshot_semantics: Literal["daily_close", "point_in_time", "latest_observed"]
    on_hand_quantity: Decimal | None = None
    reserved_quantity: Decimal | None = None
    available_quantity: Decimal | None = None
    quantity_unit: str = Field(min_length=1, max_length=64)

    @field_validator("snapshot_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("snapshot_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_quantities(self) -> "InventorySnapshotInput":
        values = (self.on_hand_quantity, self.reserved_quantity, self.available_quantity)
        if self.on_hand_quantity is None and self.available_quantity is None:
            raise ValueError("on_hand_quantity or available_quantity is required; reserved quantity alone is not a stock measure")
        for value in values:
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError("inventory quantities must be finite and non-negative")
        return self


class InventorySnapshotBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_system: str = Field(min_length=1, max_length=128)
    source_batch_id: str = Field(min_length=1, max_length=128)
    rows: list[InventorySnapshotInput] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def require_single_seller(self) -> "InventorySnapshotBatch":
        if len({row.seller_id for row in self.rows}) != 1:
            raise ValueError("one inventory batch may contain rows for only one seller")
        return self


class InventorySnapshotQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seller_id: str = Field(min_length=1, max_length=128)
    as_of: datetime
    quantity_basis: Literal["on_hand", "available"]
    warehouse_id: str | None = Field(default=None, max_length=128)
    product_id: str | None = Field(default=None, max_length=128)
    limit: int = Field(default=1000, ge=1, le=10_000)

    @field_validator("as_of")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must include a timezone")
        return value


class InventorySnapshotRequest(BaseModel):
    """HTTP request contract; caller scope is bounded again by the repository."""

    model_config = ConfigDict(extra="forbid")

    seller_id: str = Field(min_length=1, max_length=128)
    start_date: date
    end_date: date
    as_of: datetime
    quantity_basis: Literal["on_hand", "available"]
    warehouse_id: str | None = Field(default=None, max_length=128)
    product_id: str | None = Field(default=None, max_length=128)
    limit: int = Field(default=1000, ge=1, le=10_000)

    @field_validator("as_of")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_scope(self) -> "InventorySnapshotRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        if (self.end_date - self.start_date).days > 366 * 3:
            raise ValueError("Inventory scope must not exceed three calendar years")
        if not self.start_date <= self.as_of.date() <= self.end_date:
            raise ValueError("as_of must be inside the requested date scope")
        return self
