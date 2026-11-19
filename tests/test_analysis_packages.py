"""Template registration checks; registration metadata does not route user text."""

from app.analysis.templates.registry import list_templates


def test_analysis_packages_are_registered_without_natural_language_triggers() -> None:
    names = {str(item["name"]) for item in list_templates()}
    assert {
        "growth_diagnostics",
        "fulfillment_funnel",
        "customer_lifecycle",
        "category_portfolio",
        "logistics_experience",
        "payment_cancellation",
    } <= names
    assert all("trigger_phrases" not in item for item in list_templates())
