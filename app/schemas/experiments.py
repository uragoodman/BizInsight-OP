from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, model_validator


class CausalExperimentScopeRequest(BaseModel):
    seller_id: str = Field(min_length=1, max_length=64)
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_range(self) -> "CausalExperimentScopeRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        if (self.end_date - self.start_date).days > 366 * 3:
            raise ValueError("Experiment analysis date range must not exceed three calendar years")
        return self


class CausalExperimentEstimateRequest(CausalExperimentScopeRequest):
    experiment_id: str = Field(min_length=1, max_length=128)
