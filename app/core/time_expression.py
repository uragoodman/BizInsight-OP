"""聊天与数据浏览页共用的可信日历解析器。

LLM 只负责识别时间意图；只有本模块可以把意图转换成最终日期。
所有相对时间均以数据集最新可用日期为锚点。
"""
from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from sqlalchemy import text

from app.data.database import engine

MAX_RANGE_DAYS = 366 * 3

# Keep explicit ISO-day recognition tolerant of adjacent Chinese text while
# still preventing a date from being cut out of a longer numeric identifier.
# Several user-facing surfaces parse the same canonical date form; ``\b`` is
# not suitable here because CJK characters are word characters to Python's
# Unicode regular-expression engine.
_ISO_DAY_PATTERN = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)")


def extract_iso_dates(expression: str) -> list[str]:
    """Extract canonical ISO calendar-date tokens from user text.

    This function only recognizes tokens.  Validation and range semantics stay
    in :func:`resolve_expression` and the shared context date-role parser.
    """

    return _ISO_DAY_PATTERN.findall(expression or "")


@dataclass(frozen=True)
class ResolvedTimeRange:
    requested_start: date
    requested_end: date
    start_date: date
    end_date: date
    label: str
    data_min_date: date
    data_max_date: date
    truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_start": self.requested_start.isoformat(), "requested_end": self.requested_end.isoformat(),
            "start_date": self.start_date.isoformat(), "end_date": self.end_date.isoformat(), "label": self.label,
            "data_min_date": self.data_min_date.isoformat(), "data_max_date": self.data_max_date.isoformat(),
            "truncated": self.truncated,
        }


def data_time_bounds() -> tuple[date, date]:
    with engine.connect() as conn:
        # Explorer 的实际数据源是 v_delivered_orders（仅已交付订单），
        # daily_metrics 的 max 日期包含尚未交付的订单创建记录，两者
        # 存在约 49 天落差（daily_metrics 到 2018-10-17，交付数据到
        # 2018-08-29）。这里使用交付数据源，使时间预设能与 Explorer
        # 的实际数据范围对齐。
        row = conn.execute(text("SELECT MIN(purchase_date) AS first_date, MAX(purchase_date) AS last_date FROM analytics.v_delivered_orders")).mappings().one()
    if not row["first_date"] or not row["last_date"]:
        raise ValueError("No data is available to resolve a time range")
    return row["first_date"], row["last_date"]


def _finish(
    start: date,
    end: date,
    label: str,
    *,
    data_bounds: tuple[date, date] | None = None,
) -> ResolvedTimeRange:
    if end < start:
        raise ValueError("end date must not be earlier than start date")
    if (end - start).days > MAX_RANGE_DAYS:
        raise ValueError("time range must not exceed three calendar years")
    first, latest = data_bounds or data_time_bounds()
    actual_start, actual_end = max(start, first), min(end, latest)
    if actual_end < actual_start:
        raise ValueError(f"Requested {label} has no data in the available range {first} to {latest}")
    return ResolvedTimeRange(start, end, actual_start, actual_end, label, first, latest, actual_start != start or actual_end != end)


def _month_range(ref: date) -> tuple[date, date]:
    return ref.replace(day=1), ref.replace(day=monthrange(ref.year, ref.month)[1])


_ENGLISH_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_ENGLISH_MONTH_PATTERN = "|".join(sorted(_ENGLISH_MONTHS, key=len, reverse=True))


def ambiguous_year_list(question: str) -> tuple[int, ...] | None:
    """Return separately listed years that need an intent confirmation."""
    years = tuple(dict.fromkeys(int(value) for value in re.findall(r"(?<!\d)(20\d{2})\s*\u5e74", question)))
    if len(years) < 2:
        return None
    has_list_separator = bool(re.search(r"[、，,/]|\band\b|以及|和", question, re.I))
    has_intent = bool(re.search(r"趋势|trend|按年|年度|同比|环比|对比|比较|compare", question, re.I))
    return years if has_list_separator and not has_intent else None


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def _finish_month_range(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    label: str,
    *,
    data_bounds: tuple[date, date] | None = None,
) -> ResolvedTimeRange:
    start, _ = _month_bounds(start_year, start_month)
    _, end = _month_bounds(end_year, end_month)
    return _finish(start, end, label, data_bounds=data_bounds)


def resolve_preset(kind: Literal["month", "previous_month", "recent_months", "year", "half", "quarter"], *, year: int | None = None, value: int | None = None) -> ResolvedTimeRange:
    _, latest = data_time_bounds()
    if kind == "month":
        start, end = _month_range(latest); return _finish(start, end, "本月")
    if kind == "previous_month":
        start, end = _month_range(latest.replace(day=1) - timedelta(days=1)); return _finish(start, end, "上月")
    if kind == "recent_months":
        months = value or 1
        if not 1 <= months <= 36: raise ValueError("recent months must be between 1 and 36")
        month_index = latest.year * 12 + latest.month - months
        start = date(month_index // 12, month_index % 12 + 1, 1)
        return _finish(start, latest, f"最近 {months} 个月")
    target_year = year or latest.year
    if kind == "year": return _finish(date(target_year, 1, 1), date(target_year, 12, 31), f"{target_year} 年")
    if kind == "half":
        half = value or 1
        if half not in (1, 2): raise ValueError("half must be 1 or 2")
        return _finish(date(target_year, 1 if half == 1 else 7, 1), date(target_year, 6 if half == 1 else 12, 30 if half == 1 else 31), f"{target_year} {'上半年' if half == 1 else '下半年'}")
    quarter = value or 1
    if quarter not in (1, 2, 3, 4): raise ValueError("quarter must be 1 to 4")
    month = (quarter - 1) * 3 + 1
    return _finish(date(target_year, month, 1), date(target_year, month + 2, monthrange(target_year, month + 2)[1]), f"{target_year} Q{quarter}")


def resolve_expression(
    question: str,
    *,
    data_bounds: tuple[date, date] | None = None,
) -> ResolvedTimeRange | None:
    """解析中英文时间表达；可使用调用方可信的数据范围避免全局 DB 锚点。

    Scoped 查询应传入其已授权的边界。相对年份/月/日以该边界的结束日
    为锚点，解析结果也只会被裁剪到该范围内。
    """
    bounds = data_bounds if data_bounds is not None else data_time_bounds()

    def finish(start: date, end: date, label: str) -> ResolvedTimeRange:
        return _finish(start, end, label, data_bounds=bounds)

    def finish_month_range(start_year: int, start_month: int, end_year: int, end_month: int, label: str) -> ResolvedTimeRange:
        return _finish_month_range(
            start_year, start_month, end_year, end_month, label, data_bounds=bounds,
        )

    latest = bounds[1]
    dates = extract_iso_dates(question)
    if len(dates) >= 2:
        return finish(date.fromisoformat(dates[0]), date.fromisoformat(dates[1]), f"{dates[0]} 至 {dates[1]}")
    if len(dates) == 1:
        d = date.fromisoformat(dates[0]); return finish(d, d, dates[0])

    # All partial expressions inherit only a year, never a month or day, from
    # the dataset anchor.  A bare "1-30日" remains intentionally unresolved.
    latest_year = latest.year

    # Year ranges: 2017年至2018年 / from 2017 to 2018 / 2017-2018.
    year_range = re.search(
        r"(?:\u4ece\s*|from\s+)?(?P<sy>20\d{2})\s*(?:\u5e74)?\s*"
        r"(?:\u5230|\u81f3|to|through|until|[-~—–])\s*(?P<ey>20\d{2})\s*(?:\u5e74)?",
        question, re.I,
    )
    if year_range:
        return finish(date(int(year_range["sy"]), 1, 1), date(int(year_range["ey"]), 12, 31), year_range.group(0))

    # Explicit yearly trend: 2016年、2017年、2018年趋势.
    # Resolve it as one annual series, not an accidental single-year match.
    listed_years = tuple(dict.fromkeys(int(value) for value in re.findall(r"(?<!\d)(20\d{2})\s*\u5e74", question)))
    if (
        len(listed_years) >= 2
        and re.search(r"[、，,/]|\band\b|以及|和", question, re.I)
        and re.search(r"趋势|trend|按年|年度", question, re.I)
    ):
        start_year, end_year = min(listed_years), max(listed_years)
        return finish(date(start_year, 1, 1), date(end_year, 12, 31), "、".join(f"{year}年" for year in listed_years))

    # Explicit Chinese dates: 2018年1月1日到5月31日 / 2018年1月1日至2018年5月31日.
    chinese_dates = re.search(
        r"(?P<sy>20\d{2})\s*\u5e74\s*(?P<sm>\d{1,2})\s*\u6708\s*(?P<sd>\d{1,2})\s*(?:\u65e5)?"
        r"\s*(?:\u5230|\u81f3|to|[-~—–])\s*(?:(?P<ey>20\d{2})\s*\u5e74\s*)?"
        r"(?P<em>\d{1,2})\s*\u6708\s*(?P<ed>\d{1,2})\s*(?:\u65e5)?",
        question, re.I,
    )
    if chinese_dates:
        start = date(int(chinese_dates["sy"]), int(chinese_dates["sm"]), int(chinese_dates["sd"]))
        end = date(int(chinese_dates["ey"] or chinese_dates["sy"]), int(chinese_dates["em"]), int(chinese_dates["ed"]))
        return finish(start, end, chinese_dates.group(0))

    # Chinese date ranges with an omitted end month/year: 2018年5月1日至30日.
    chinese_same_month_dates = re.search(
        r"(?P<sy>20\d{2})\s*\u5e74\s*(?P<sm>\d{1,2})\s*\u6708\s*(?P<sd>\d{1,2})\s*(?:\u65e5|\u53f7)?"
        r"\s*(?:\u5230|\u81f3|to|[-~—–])\s*(?P<ed>\d{1,2})\s*(?:\u65e5|\u53f7)",
        question, re.I,
    )
    if chinese_same_month_dates:
        start = date(int(chinese_same_month_dates["sy"]), int(chinese_same_month_dates["sm"]), int(chinese_same_month_dates["sd"]))
        end = date(int(chinese_same_month_dates["sy"]), int(chinese_same_month_dates["sm"]), int(chinese_same_month_dates["ed"]))
        return finish(start, end, chinese_same_month_dates.group(0))

    # A month/day range without year is safe only because the month token is explicit.
    chinese_partial_dates = re.search(
        r"(?P<sm>\d{1,2})\s*\u6708\s*(?P<sd>\d{1,2})\s*(?:\u65e5|\u53f7)?"
        r"\s*(?:\u5230|\u81f3|to|[-~—–])\s*(?:(?P<em>\d{1,2})\s*\u6708\s*)?(?P<ed>\d{1,2})\s*(?:\u65e5|\u53f7)",
        question, re.I,
    )
    if chinese_partial_dates:
        start = date(latest_year, int(chinese_partial_dates["sm"]), int(chinese_partial_dates["sd"]))
        end = date(latest_year, int(chinese_partial_dates["em"] or chinese_partial_dates["sm"]), int(chinese_partial_dates["ed"]))
        return finish(start, end, chinese_partial_dates.group(0))

    chinese_date = re.search(r"(?P<year>20\d{2})\s*\u5e74\s*(?P<month>\d{1,2})\s*\u6708\s*(?P<day>\d{1,2})\s*(?:\u65e5)?", question)
    if chinese_date:
        value = date(int(chinese_date["year"]), int(chinese_date["month"]), int(chinese_date["day"]))
        return finish(value, value, chinese_date.group(0))

    english_dates = re.search(
        rf"(?P<sm>{_ENGLISH_MONTH_PATTERN})\.?\s+(?P<sd>\d{{1,2}}),?\s*(?P<sy>20\d{{2}})\s*"
        rf"(?:to|through|until|[-~—–])\s*(?P<em>{_ENGLISH_MONTH_PATTERN})\.?\s+(?P<ed>\d{{1,2}}),?\s*(?P<ey>20\d{{2}})?",
        question, re.I,
    )
    if english_dates:
        start = date(int(english_dates["sy"]), _ENGLISH_MONTHS[english_dates["sm"].lower()], int(english_dates["sd"]))
        end = date(int(english_dates["ey"] or english_dates["sy"]), _ENGLISH_MONTHS[english_dates["em"].lower()], int(english_dates["ed"]))
        return finish(start, end, english_dates.group(0))

    english_partial_dates = re.search(
        rf"(?P<sm>{_ENGLISH_MONTH_PATTERN})\.?\s+(?P<sd>\d{{1,2}})\s*(?:to|through|until|[-~—–])\s*"
        rf"(?:(?P<em>{_ENGLISH_MONTH_PATTERN})\.?\s+)?(?P<ed>\d{{1,2}})(?:,?\s*(?P<year>20\d{{2}}))?",
        question, re.I,
    )
    if english_partial_dates:
        year = int(english_partial_dates["year"] or latest_year)
        start = date(year, _ENGLISH_MONTHS[english_partial_dates["sm"].lower()], int(english_partial_dates["sd"]))
        end = date(year, _ENGLISH_MONTHS[(english_partial_dates["em"] or english_partial_dates["sm"]).lower()], int(english_partial_dates["ed"]))
        return finish(start, end, english_partial_dates.group(0))

    english_date = re.search(rf"(?P<month>{_ENGLISH_MONTH_PATTERN})\.?\s+(?P<day>\d{{1,2}}),?\s*(?P<year>20\d{{2}})", question, re.I)
    if english_date:
        value = date(int(english_date["year"]), _ENGLISH_MONTHS[english_date["month"].lower()], int(english_date["day"]))
        return finish(value, value, english_date.group(0))

    # Chinese month ranges: 2018年1月到5月, 2018年1-5月, 1月至5月 2018年.
    chinese_months = re.search(
        r"(?P<year>20\d{2})\s*\u5e74\s*(?P<sm>\d{1,2})\s*\u6708?\s*"
        r"(?:\u5230|\u81f3|to|[-~—–])\s*(?:(?P<ey>20\d{2})\s*\u5e74\s*)?"
        r"(?P<em>\d{1,2})\s*\u6708",
        question, re.I,
    )
    if not chinese_months:
        chinese_months = re.search(
            r"(?P<sm>\d{1,2})\s*\u6708?\s*(?:\u5230|\u81f3|to|[-~—–])\s*(?P<em>\d{1,2})\s*\u6708\s*(?P<year>20\d{2})\s*\u5e74",
            question, re.I,
        )
    if not chinese_months:
        chinese_months = re.search(
            r"(?P<sm>\d{1,2})\s*\u6708?\s*(?:\u5230|\u81f3|to|[-~—–])\s*(?P<em>\d{1,2})\s*\u6708",
            question, re.I,
        )
    if chinese_months:
        year = int(chinese_months.groupdict().get("year") or latest_year)
        return finish_month_range(year, int(chinese_months["sm"]), int(chinese_months.groupdict().get("ey") or year), int(chinese_months["em"]), chinese_months.group(0))

    # English month ranges: Jan-May 2018 / January to May 2018.
    english_months = re.search(
        rf"(?P<sm>{_ENGLISH_MONTH_PATTERN})\.?\s*(?:to|through|until|[-~—–])\s*(?P<em>{_ENGLISH_MONTH_PATTERN})\.?\s*,?\s*(?P<year>20\d{{2}})?",
        question, re.I,
    )
    if english_months:
        return finish_month_range(
            int(english_months["year"] or latest_year), _ENGLISH_MONTHS[english_months["sm"].lower()],
            int(english_months["year"] or latest_year), _ENGLISH_MONTHS[english_months["em"].lower()], english_months.group(0),
        )

    # A single explicit month, Chinese or English. A month without a year uses the dataset's latest year.
    chinese_month = re.search(r"(?:(?P<year>20\d{2})\s*\u5e74\s*)?(?P<month>\d{1,2})\s*\u6708", question)
    if chinese_month:
        year = int(chinese_month["year"] or latest_year)
        start, end = _month_bounds(year, int(chinese_month["month"]))
        return finish(start, end, chinese_month.group(0))
    english_month = re.search(rf"(?P<month>{_ENGLISH_MONTH_PATTERN})\.?\s*(?P<year>20\d{{2}})?", question, re.I)
    if english_month:
        year = int(english_month["year"] or latest_year)
        start, end = _month_bounds(year, _ENGLISH_MONTHS[english_month["month"].lower()])
        return finish(start, end, english_month.group(0))
    match = re.search(r"(?:最近|recent)\s*(\d+)\s*(?:个月|月|months?)", question, re.I)
    if match:
        months = int(match.group(1))
        if not 1 <= months <= 36:
            raise ValueError("recent months must be between 1 and 36")
        month_index = latest.year * 12 + latest.month - months
        start = date(month_index // 12, month_index % 12 + 1, 1)
        return finish(start, latest, f"最近 {months} 个月")
    match = re.search(r"(?:最近|recent)\s*(\d+)\s*(?:天|days?)", question, re.I)
    if match:
        days = int(match.group(1))
        if not 1 <= days <= MAX_RANGE_DAYS: raise ValueError("recent days must be within three calendar years")
        return finish(latest - timedelta(days=days - 1), latest, f"最近 {days} 天")
    latest_year = latest.year
    year_match = re.search(r"\b(20\d{2})\s*(?:年)?", question)
    year = int(year_match.group(1)) if year_match else latest_year
    quarter = re.search(r"(?:\bQ([1-4])\b|([1-4])\s*季度)", question, re.I)
    if quarter:
        q = int(quarter.group(1) or quarter.group(2))
        month = (q - 1) * 3 + 1
        return finish(date(year, month, 1), date(year, month + 2, monthrange(year, month + 2)[1]), f"{year} Q{q}")
    if "上半年" in question: return finish(date(year, 1, 1), date(year, 6, 30), f"{year} 上半年")
    if "下半年" in question: return finish(date(year, 7, 1), date(year, 12, 31), f"{year} 下半年")
    if re.search(r"\b20\d{2}\s*年", question): return finish(date(year, 1, 1), date(year, 12, 31), f"{year} 年")
    if "上月" in question or "上个月" in question:
        start, end = _month_range(latest.replace(day=1) - timedelta(days=1))
        return finish(start, end, "上月")
    if "本月" in question:
        start, end = _month_range(latest)
        return finish(start, end, "本月")
    return None
