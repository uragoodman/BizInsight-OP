from pydantic import BaseModel


class MetricPoint(BaseModel):
    date: str
    value: float
