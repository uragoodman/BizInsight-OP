"""Bounded metadata profiling for uploaded XLSX and Parquet snapshots."""

from __future__ import annotations

import io
import math
import re
import zipfile
from datetime import date, datetime
from typing import Any


SENSITIVE_FIELD = re.compile(
    r"(?:^|[_\W])(?:email|e-mail|phone|mobile|address|cpf|tax[_-]?id|"
    r"customer[_-]?(?:unique[_-]?)?id|user[_-]?id|account[_-]?id|"
    r"credit[_-]?card|password|token|secret|name|姓名|身份证|手机号|"
    r"电话|邮箱|地址)(?:$|[_\W])",
    re.IGNORECASE,
)
MAX_XLSX_ENTRIES = 10_000
MAX_PROFILE_ROWS = 10_000


def is_sensitive_field(name: str) -> bool:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return bool(SENSITIVE_FIELD.search(normalized))


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)[:200]


def _masked_sample(names: list[str], values: list[Any]) -> dict[str, Any]:
    return {
        name: "[masked]" if is_sensitive_field(name) else _json_scalar(values[index])
        for index, name in enumerate(names)
    }


def _unique_headers(values: list[Any], count: int) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}
    for index in range(count):
        raw = values[index] if index < len(values) else None
        base = str(raw).strip()[:128] if raw is not None else ""
        base = base or f"column_{index + 1}"
        occurrence = seen.get(base, 0) + 1
        seen[base] = occurrence
        headers.append(base if occurrence == 1 else f"{base}_{occurrence}")
    return headers


def _python_type(values: list[Any]) -> str:
    observed = {type(value) for value in values if value is not None}
    if not observed:
        return "unknown"
    if observed <= {bool}:
        return "boolean"
    if observed <= {int, float}:
        return "number"
    if observed <= {date, datetime}:
        return "datetime"
    if observed <= {str}:
        return "string"
    return "mixed"


def _validate_xlsx_archive(raw: bytes, max_uncompressed_bytes: int) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded XLSX archive is invalid") from exc
    with archive:
        members = archive.infolist()
        if len(members) > MAX_XLSX_ENTRIES:
            raise ValueError("XLSX contains too many archive entries")
        total = sum(max(0, member.file_size) for member in members)
        if total > max_uncompressed_bytes:
            raise ValueError("XLSX uncompressed content exceeds the safety limit")
        lowered = {member.filename.lower() for member in members}
        if any(
            marker in name
            for name in lowered
            for marker in (
                "vbaproject",
                "/embeddings/",
                "/oleobjects/",
                "/externallinks/",
            )
        ):
            raise ValueError("XLSX contains unsupported active or linked content")
        for member in members:
            compressed = max(1, member.compress_size)
            if member.file_size > 1_000_000 and member.file_size / compressed > 100:
                raise ValueError("XLSX contains a suspiciously compressed entry")


def xlsx_contract(raw: bytes, settings: Any) -> dict[str, Any]:
    _validate_xlsx_archive(
        raw,
        int(
            getattr(
                settings,
                "code_analysis_max_uncompressed_bytes",
                256 * 1024 * 1024,
            )
        ),
    )
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("XLSX profiling requires openpyxl") from exc
    try:
        workbook = load_workbook(
            io.BytesIO(raw),
            read_only=True,
            data_only=True,
            keep_links=False,
        )
    except Exception as exc:
        raise ValueError("XLSX workbook structure is invalid") from exc
    try:
        if len(workbook.worksheets) > 50:
            raise ValueError("XLSX contains too many worksheets")
        sheet_contracts: list[dict[str, Any]] = []
        primary_columns: list[dict[str, Any]] = []
        primary_samples: list[dict[str, Any]] = []
        total_rows = 0
        for worksheet in workbook.worksheets:
            row_count = max(int(worksheet.max_row or 0) - 1, 0)
            column_count = int(worksheet.max_column or 0)
            total_rows += row_count
            if column_count > settings.code_analysis_max_columns:
                raise ValueError("XLSX exceeds the column limit")
            if total_rows > settings.code_analysis_max_rows:
                raise ValueError("XLSX exceeds the row limit")
            sheet_contracts.append(
                {
                    "name": str(worksheet.title)[:128],
                    "rows": row_count,
                    "columns": column_count,
                }
            )
            if primary_columns or not column_count:
                continue
            iterator = worksheet.iter_rows(
                min_row=1,
                max_row=min(int(worksheet.max_row or 0), MAX_PROFILE_ROWS + 1),
                max_col=column_count,
                values_only=True,
            )
            header_values = list(next(iterator, ()))
            names = _unique_headers(header_values, column_count)
            values_by_column: list[list[Any]] = [[] for _ in names]
            samples: list[dict[str, Any]] = []
            profiled_rows = 0
            for row in iterator:
                profiled_rows += 1
                values = list(row) + [None] * max(0, column_count - len(row))
                for index, value in enumerate(values[:column_count]):
                    values_by_column[index].append(value)
                if len(samples) < 5:
                    samples.append(_masked_sample(names, values))
            primary_columns = [
                {
                    "name": name,
                    "type": _python_type(values_by_column[index]),
                    "null_rate": (
                        round(
                            sum(value is None or value == "" for value in values_by_column[index])
                            / profiled_rows,
                            4,
                        )
                        if profiled_rows
                        else 0
                    ),
                    "null_rate_sampled": row_count > profiled_rows,
                    "sensitive": is_sensitive_field(name),
                }
                for index, name in enumerate(names)
            ]
            primary_samples = samples
        return {
            "file_type": "xlsx",
            "rows": total_rows,
            "columns": primary_columns,
            "samples": primary_samples,
            "sheets": sheet_contracts,
            "profiled_rows": min(total_rows, MAX_PROFILE_ROWS),
            "warnings": [
                "Workbook formulas and links are never executed; cached cell values are read as data only.",
                "Column quality statistics are sampled when a sheet exceeds 10,000 rows.",
            ],
        }
    finally:
        workbook.close()


def parquet_contract(raw: bytes, settings: Any) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("Parquet profiling requires pyarrow") from exc
    try:
        parquet = pq.ParquetFile(io.BytesIO(raw))
        schema = parquet.schema_arrow
        metadata = parquet.metadata
    except Exception as exc:
        raise ValueError("Parquet metadata is invalid") from exc
    rows = int(metadata.num_rows)
    names = list(schema.names)
    if rows > settings.code_analysis_max_rows:
        raise ValueError("Parquet exceeds the row limit")
    if len(names) == 0 or len(names) > settings.code_analysis_max_columns:
        raise ValueError("Parquet has no columns or exceeds the column limit")
    for field in schema:
        if (
            pa.types.is_struct(field.type)
            or pa.types.is_list(field.type)
            or pa.types.is_large_list(field.type)
            or pa.types.is_map(field.type)
        ):
            raise ValueError("Nested Parquet columns are not supported in the first release")
    null_counts = [0] * len(names)
    minima: list[Any] = [None] * len(names)
    maxima: list[Any] = [None] * len(names)
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for index in range(min(row_group.num_columns, len(names))):
            statistics = row_group.column(index).statistics
            if not statistics:
                continue
            if statistics.null_count is not None:
                null_counts[index] += int(statistics.null_count)
            if statistics.has_min_max:
                if minima[index] is None:
                    minima[index] = statistics.min
                    maxima[index] = statistics.max
                else:
                    try:
                        minima[index] = min(minima[index], statistics.min)
                        maxima[index] = max(maxima[index], statistics.max)
                    except TypeError:
                        minima[index] = maxima[index] = None
    samples: list[dict[str, Any]] = []
    if rows:
        try:
            batch = next(parquet.iter_batches(batch_size=5, columns=names))
            for record in batch.to_pylist():
                values = [record.get(name) for name in names]
                samples.append(_masked_sample(names, values))
        except StopIteration:
            pass
        except Exception as exc:
            raise ValueError("Parquet sample rows could not be decoded") from exc
    columns = [
        {
            "name": name,
            "type": str(schema.field(name).type),
            "null_rate": round(null_counts[index] / rows, 4) if rows else 0,
            "sensitive": is_sensitive_field(name),
            "min": _json_scalar(minima[index]),
            "max": _json_scalar(maxima[index]),
        }
        for index, name in enumerate(names)
    ]
    return {
        "file_type": "parquet",
        "rows": rows,
        "columns": columns,
        "samples": samples,
        "row_groups": int(metadata.num_row_groups),
        "warnings": [
            "Only metadata and a five-row sample are read before confirmation.",
        ],
    }


def binary_contract(file_type: str, raw: bytes, settings: Any) -> dict[str, Any]:
    if file_type == "xlsx":
        return xlsx_contract(raw, settings)
    if file_type == "parquet":
        return parquet_contract(raw, settings)
    raise ValueError("Unsupported binary analysis file type")
