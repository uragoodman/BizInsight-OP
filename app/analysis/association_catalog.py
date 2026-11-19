"""Server-owned association pair registry; never populated from model output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class AssociationPair:
    pair_id: str
    target_metric_id: str
    kind: Literal["binary", "continuous"]
    exposure_label: str
    outcome_label: str
    method_id: str
    minimum_sample_size: int
    query_handler: str
    source_table: str
    analysis_unit: str
    population_definition: str
    eligibility_definition: str
    exclusion_policy: str
    limitations: tuple[str, ...]

    def evidence_contract(self) -> dict[str, object]:
        """Return the immutable server-owned statistical contract for Evidence."""
        return {
            "method_id": self.method_id,
            "source_table": self.source_table,
            "analysis_unit": self.analysis_unit,
            "population_definition": self.population_definition,
            "exposure_definition": self.exposure_label,
            "outcome_definition": self.outcome_label,
            "eligibility_definition": self.eligibility_definition,
            "exclusion_policy": self.exclusion_policy,
            "minimum_sample_size": self.minimum_sample_size,
        }


_PAIRS = {
    "delivery_timing_bad_review_v1": AssociationPair(
        pair_id="delivery_timing_bad_review_v1",
        target_metric_id="seller_order_bad_review_rate",
        kind="binary",
        exposure_label="是否准时送达（is_on_time）",
        outcome_label="是否低评分（review_score <= 2）",
        method_id="binary_risk_difference_newcombe_v1",
        minimum_sample_size=50,
        query_handler="delivery_timing_bad_review",
        source_table="analytics.v_order_unified",
        analysis_unit="single_seller_order",
        population_definition="Single-seller orders in the authorized purchase-date range.",
        eligibility_definition=(
            "review_score and is_on_time are both non-null; low rating is review_score <= 2."
        ),
        exclusion_policy=(
            "Orders missing review_score or is_on_time are excluded from the 2x2 analysis and counted separately."
        ),
        limitations=(
            "No confounder adjustment; association does not establish causality.",
            "At most one registered pair is analyzed per investigation; no multiplicity correction is applied.",
        ),
    ),
    "delivery_hours_review_score_spearman_v1": AssociationPair(
        pair_id="delivery_hours_review_score_spearman_v1",
        target_metric_id="review_score_avg",
        kind="continuous",
        exposure_label="实际配送时长（小时）",
        outcome_label="订单评分（1–5）",
        method_id="spearman_rank_correlation_fisher_approx_v1",
        minimum_sample_size=50,
        query_handler="delivery_hours_review_score",
        source_table="analytics.v_order_unified",
        analysis_unit="single_seller_order",
        population_definition="Single-seller orders in the authorized purchase-date range.",
        eligibility_definition=(
            "review_score and non-negative delivery_transit_hours are both present."
        ),
        exclusion_policy=(
            "Orders missing review_score, missing delivery duration, or with negative duration are excluded and counted separately."
        ),
        limitations=(
            "Spearman describes monotonic association, not arbitrary nonlinear dependence.",
            "No confounder adjustment; association does not establish causality.",
            "Association does not establish causality.",
            "The interval and p value use large-sample Fisher/normal approximations.",
            "At most one registered pair is analyzed per investigation; no multiplicity correction is applied.",
        ),
    ),
    "order_freight_review_score_spearman_v1": AssociationPair(
        pair_id="order_freight_review_score_spearman_v1",
        target_metric_id="review_score_avg",
        kind="continuous",
        exposure_label="单店订单运费总额（freight_sum）",
        outcome_label="订单评分（review_score，1–5）",
        method_id="spearman_rank_correlation_fisher_approx_v1",
        minimum_sample_size=50,
        query_handler="order_freight_review_score",
        source_table="analytics.v_order_unified",
        analysis_unit="single_seller_order",
        population_definition="Single-seller orders in the authorized purchase-date range.",
        eligibility_definition=(
            "review_score and non-negative order freight total (freight_sum) are both present."
        ),
        exclusion_policy=(
            "Orders missing review_score, missing freight_sum, or with negative freight_sum are excluded and counted separately."
        ),
        limitations=(
            "Spearman describes monotonic association, not arbitrary nonlinear dependence.",
            "No confounder adjustment; association does not establish causality.",
            "Freight may co-vary with delivery distance and order/product composition.",
            "The interval and p value use large-sample Fisher/normal approximations.",
            "At most one registered pair is analyzed per investigation; no multiplicity correction is applied.",
        ),
    ),
    "order_item_line_count_review_score_spearman_v1": AssociationPair(
        pair_id="order_item_line_count_review_score_spearman_v1",
        target_metric_id="review_score_avg",
        kind="continuous",
        exposure_label="单店订单商品明细行数（item_count；不是商品件数）",
        outcome_label="订单评分（review_score，1–5）",
        method_id="spearman_rank_correlation_fisher_approx_v1",
        minimum_sample_size=50,
        query_handler="order_item_line_count_review_score",
        source_table="analytics.v_order_unified",
        analysis_unit="single_seller_order",
        population_definition="Single-seller orders in the authorized purchase-date range.",
        eligibility_definition=(
            "review_score and non-negative item line count (item_count) are both present."
        ),
        exclusion_policy=(
            "Orders missing review_score, missing item_count, or with negative item_count are excluded and counted separately."
        ),
        limitations=(
            "item_count counts order-item rows, not the quantity of units purchased.",
            "Spearman describes monotonic association, not arbitrary nonlinear dependence.",
            "No confounder adjustment; association does not establish causality.",
            "The interval and p value use large-sample Fisher/normal approximations.",
            "At most one registered pair is analyzed per investigation; no multiplicity correction is applied.",
        ),
    ),
    "order_merchandise_value_review_score_spearman_v1": AssociationPair(
        pair_id="order_merchandise_value_review_score_spearman_v1",
        target_metric_id="review_score_avg",
        kind="continuous",
        exposure_label="单店订单商品金额（price_sum；BRL，不含运费）",
        outcome_label="订单评分（review_score，1–5）",
        method_id="spearman_rank_correlation_fisher_approx_v1",
        minimum_sample_size=50,
        query_handler="order_merchandise_value_review_score",
        source_table="analytics.v_order_unified",
        analysis_unit="single_seller_order",
        population_definition="Single-seller orders in the authorized purchase-date range.",
        eligibility_definition=(
            "review_score and non-negative merchandise subtotal (price_sum, excluding freight) are both present."
        ),
        exclusion_policy=(
            "Orders missing review_score, missing price_sum, or with negative price_sum are excluded and counted separately."
        ),
        limitations=(
            "Spearman describes monotonic association, not arbitrary nonlinear dependence.",
            "No confounder adjustment; association does not establish causality.",
            "Merchandise value may co-vary with item count and product/order composition.",
            "The interval and p value use large-sample Fisher/normal approximations.",
            "At most one registered pair is analyzed per investigation; no multiplicity correction is applied.",
        ),
    ),
}


def get_association_pair(pair_id: str) -> AssociationPair | None:
    return _PAIRS.get(pair_id)


def registered_pair_ids_for_metric(metric_id: str) -> tuple[str, ...]:
    return tuple(sorted(pair_id for pair_id, pair in _PAIRS.items() if pair.target_metric_id == metric_id))


def is_registered_pair_for_metric(pair_id: str, metric_id: str) -> bool:
    pair = get_association_pair(pair_id)
    return pair is not None and pair.target_metric_id == metric_id
