"""Bounded upload-contract profiling tests."""

from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace

import pytest

from app.code_analysis.profiling import parquet_contract, xlsx_contract


SETTINGS = SimpleNamespace(
    code_analysis_max_rows=1_000,
    code_analysis_max_columns=20,
)


def test_xlsx_contract_masks_sensitive_values_and_does_not_execute_formulas() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    sheet.append(["month", "gmv", "手机号"])
    sheet.append(["2024-01", 12.5, "phone-sample-a"])
    sheet.append(["2024-02", "=10+5", "phone-sample-b"])
    payload = io.BytesIO()
    workbook.save(payload)
    workbook.close()

    contract = xlsx_contract(payload.getvalue(), SETTINGS)
    assert contract["rows"] == 2
    assert contract["sheets"] == [{"name": "Sales", "rows": 2, "columns": 3}]
    assert contract["columns"][1]["type"] in {"number", "mixed"}
    assert contract["columns"][2]["sensitive"] is True
    assert contract["samples"][0]["手机号"] == "[masked]"
    assert all(
        not str(sample["gmv"]).startswith("=")
        for sample in contract["samples"]
    )


def test_xlsx_contract_rejects_suspicious_compression_before_xml_parsing() -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", b"x" * 2_000_000)
    with pytest.raises(ValueError, match="suspiciously compressed"):
        xlsx_contract(payload.getvalue(), SETTINGS)


def test_parquet_contract_reads_metadata_and_masks_samples() -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "month": ["2024-01", "2024-02"],
            "gmv": [10.0, 15.0],
            "email": ["a@example.com", "b@example.com"],
        }
    )
    payload = io.BytesIO()
    pq.write_table(table, payload)

    contract = parquet_contract(payload.getvalue(), SETTINGS)
    assert contract["rows"] == 2
    assert contract["columns"][1]["name"] == "gmv"
    assert contract["columns"][2]["sensitive"] is True
    assert contract["samples"][0]["email"] == "[masked]"
