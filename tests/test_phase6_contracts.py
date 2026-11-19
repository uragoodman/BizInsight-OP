"""Data contract tests for Phase 6 seed data integrity.

These tests verify the CI seed script produces the expected data shape,
enabling downstream M1-M4 code to rely on well-known column and row guarantees.
They are designed to pass against both fresh CI seeds and the production
database (which has richer, real-world data).
"""

from app.data.database import engine
from sqlalchemy import text


def test_raw_order_reviews_has_text_columns() -> None:
    """raw.order_reviews DDL must include review_comment_message. (seed_phase6_ci_postgres.py:44-50)"""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='raw' AND table_name='order_reviews'
              AND column_name IN ('review_comment_title', 'review_comment_message',
                                  'review_creation_date', 'review_answer_timestamp')
        """)).mappings().all()
    cols = {r["column_name"]: r["data_type"] for r in rows}
    assert "review_comment_message" in cols, "review_comment_message column missing"
    assert cols["review_comment_message"] in ("text", "character varying"), (
        f"review_comment_message type {cols.get('review_comment_message')} not text"
    )


def test_raw_category_translation_exists() -> None:
    """raw.product_category_name_translation must exist and have rows. (DDL line 57-60)"""
    with engine.connect() as conn:
        count = conn.execute(text(
            "SELECT COUNT(*) FROM raw.product_category_name_translation"
        )).scalar()
    assert count >= 5, f"expected >=5 translations, got {count}"


def test_raw_has_multi_seller_order() -> None:
    """At least one order has order_items from 2+ different sellers. (SEED line 111-114)"""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT order_id, COUNT(DISTINCT seller_id) AS seller_count
            FROM raw.order_items
            GROUP BY order_id
            HAVING COUNT(DISTINCT seller_id) >= 2
            LIMIT 1
        """)).mappings().all()
    assert len(rows) >= 1, (
        "no order with 2+ sellers found — multi-seller fixture may be missing"
    )


def test_review_comment_text_has_null_and_nonnull() -> None:
    """review_comment_message has both NULL and non-NULL values. (SEED line 117-119)"""
    with engine.connect() as conn:
        has_null = conn.execute(text(
            "SELECT COUNT(*) FROM raw.order_reviews WHERE review_comment_message IS NULL"
        )).scalar()
        has_text = conn.execute(text(
            "SELECT COUNT(*) FROM raw.order_reviews WHERE review_comment_message IS NOT NULL"
        )).scalar()
    assert has_null > 0, "all reviews would have text — seed missing NULL values"
    assert has_text > 0, "no reviews would have text — seed missing non-NULL values"


def test_review_comment_rate_excludes_empty_on_aggregate() -> None:
    """含文本评论率应正确排除空文本记录。"""
    with engine.connect() as conn:
        total = conn.execute(text(
            "SELECT COUNT(*) FROM raw.order_reviews"
        )).scalar()
        with_text = conn.execute(text(
            "SELECT COUNT(*) FROM raw.order_reviews "
            "WHERE review_comment_message IS NOT NULL AND TRIM(review_comment_message) != ''"
        )).scalar()
    assert total > 0
    assert with_text < total, "comment rate should be < 100% when NULLs exist"
    rate = with_text * 100.0 / total
    assert rate > 0, "comment rate should be > 0 when text values exist"
