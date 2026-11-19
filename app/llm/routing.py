"""Deterministic profile routing; models never choose their own permissions."""
from __future__ import annotations
from app.core.config import get_settings

def profile_for(operation: str, *, quality: bool = False) -> str:
    if quality or operation in {"report", "long_answer"}: return "quality"
    if operation in {"classify", "intent", "extract"}: return "fast"
    return "standard"

def model_for(operation: str, explicit: str | None = None, *, quality: bool = False) -> tuple[str,str]:
    if explicit: return explicit, "explicit"
    settings=get_settings(); profile=profile_for(operation,quality=quality)
    configured={"fast":settings.llm_fast_model,"standard":settings.llm_standard_model,"quality":settings.llm_quality_model}[profile]
    return configured or settings.llm_model, profile

def price_for(profile: str) -> tuple[float,float]:
    s=get_settings()
    if profile=="fast": return s.llm_fast_input_cost_per_1k or s.llm_input_cost_per_1k,s.llm_fast_output_cost_per_1k or s.llm_output_cost_per_1k
    if profile=="quality": return s.llm_quality_input_cost_per_1k or s.llm_input_cost_per_1k,s.llm_quality_output_cost_per_1k or s.llm_output_cost_per_1k
    return s.llm_input_cost_per_1k,s.llm_output_cost_per_1k
