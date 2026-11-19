import copy

import pytest

from app.agents.critic_agent import review_data_query_result, review_data_query_item


def contribution():
    return {'status': 'ok', 'parsed': {'query_type': 'metric_contribution', 'metric_name': 'freight_value',
                                      'authorized_metric': {'metric_id': 'freight_value'}},
            'contribution_result': {'status': 'ok', 'metric_name': 'freight_value',
                'baseline_total': 5667.01, 'current_total': 4905.74, 'total_delta': -761.27,
                'baseline_items': [{'dimension_value': 'A', 'value': 5667.01}],
                'current_items': [{'dimension_value': 'A', 'value': 4905.74}],
                'source_table': 'raw.order_items'}}


def test_valid_contribution_is_a_supported_result():
    review=review_data_query_result(contribution())
    assert review['status']=='approved'
    assert all(review['checks'].values())


@pytest.mark.parametrize('field,value', [('baseline_total',None),('current_total',float('nan')),
    ('total_delta',-76.73),('total_delta',True),('current_items',None),('status','not_found')])
def test_incomplete_or_inconsistent_contribution_rejected(field,value):
    result=contribution();result['contribution_result'][field]=value
    assert review_data_query_result(result)['status']=='rejected'


def test_source_and_invocation_binding_remain_required():
    inner=contribution();wrapped={'request_id':'r','evidence_id':'e','server':'s','tool_name':'compare_breakdown_by_dimension','result':inner}
    plan=inner['parsed'];calls=[{'request_id':'r','planned_metric_id':'freight_value','planned_operation':'metric_contribution',
                               'server':'s','tool_name':wrapped['tool_name'],'status':'ok'}]
    bundles=[{'evidence_id':'e','source_ref':wrapped['tool_name'],'content':{'request_id':'r','metric_id':'freight_value','query_type':'metric_contribution','result':copy.deepcopy(inner)}}]
    assert review_data_query_item(wrapped,plan,bundles,calls)['status']=='approved'
    assert review_data_query_item(wrapped,plan,[],calls)['status']=='failed'
    assert review_data_query_item(wrapped,plan,bundles,[])['status']=='failed'
    inner['contribution_result'].pop('source_table')
    bundles[0]['content']['result']=copy.deepcopy(inner)
    assert review_data_query_item(wrapped,plan,bundles,calls)['status']=='failed'


def ratio_period_contribution():
    return {
        "status": "ok",
        "parsed": {
            "query_type": "metric_contribution",
            "metric_name": "freight_rate",
            "dimension": "month_position",
            "current_start": "2017-01-01", "current_end": "2017-01-31",
            "baseline_start": "2016-12-01", "baseline_end": "2016-12-31",
            "authorized_metric": {"metric_id": "freight_rate"},
        },
        "contribution_result": {
            "metric_name": "freight_rate",
            "numerator_metric": "freight_value", "denominator_metric": "gmv",
            "dimension": "month_position",
            "baseline_period": {"start_date": "2016-12-01", "end_date": "2016-12-31"},
            "current_period": {"start_date": "2017-01-01", "end_date": "2017-01-31"},
            "baseline_groups": {"month_start": {"numerator": 2.0, "denominator": 20.0}},
            "current_groups": {"month_start": {"numerator": 3.0, "denominator": 30.0}},
            "baseline_numerator_total": 2.0, "baseline_denominator_total": 20.0,
            "current_numerator_total": 3.0, "current_denominator_total": 30.0,
            "complete": True, "source_table": "analytics.fact_orders", "status": "ok",
        },
    }


def test_registered_ratio_period_facts_are_validated_against_registry_and_reconciled():
    result = ratio_period_contribution()
    review = review_data_query_result(result)
    assert review["status"] == "approved"
    assert review["checks"]["has_result"] is True
    assert review["checks"]["has_source"] is True


def test_registered_ratio_period_facts_pass_only_with_matching_plan_call_and_evidence():
    inner = ratio_period_contribution()
    request_id = "ratio-request-1"
    tool_name = "compare_freight_rate_ratio_facts_by_dimension_period"
    wrapper = {
        "request_id": request_id, "evidence_id": "ratio-evidence-1",
        "server": "bizdata-mcp-server", "tool_name": tool_name, "result": inner,
    }
    planned = {"metric_name": "freight_rate", "query_type": "metric_contribution"}
    calls = [{
        "request_id": request_id, "planned_metric_id": "freight_rate",
        "planned_operation": "metric_contribution", "server": wrapper["server"],
        "tool_name": tool_name, "status": "ok",
    }]
    bundles = [{
        "evidence_id": wrapper["evidence_id"], "source_ref": tool_name,
        "content": {
            "request_id": request_id, "metric_id": "freight_rate",
            "query_type": "metric_contribution", "result": copy.deepcopy(inner),
        },
    }]
    assert review_data_query_item(wrapper, planned, bundles, calls)["status"] == "approved"

    calls[0]["planned_metric_id"] = "gmv"
    failed = review_data_query_item(wrapper, planned, bundles, calls)
    assert failed["status"] == "failed"
    assert failed["binding_checks"]["exact_tool_invocation_matches"] is False


@pytest.mark.parametrize("mutate", [
    lambda result: result["contribution_result"].update(complete=False),
    lambda result: result["contribution_result"].update(current_numerator_total=4.0),
    lambda result: result["contribution_result"].update(denominator_metric="order_count"),
    lambda result: result["contribution_result"]["current_period"].update(start_date="2017-02-01"),
    lambda result: result["contribution_result"].update(current_denominator_total=0.0),
    lambda result: result["contribution_result"]["current_groups"]["month_start"].update(numerator=float("nan")),
])
def test_malformed_registered_ratio_period_facts_are_rejected(mutate):
    result = ratio_period_contribution()
    mutate(result)
    review = review_data_query_result(result)
    assert review["status"] == "rejected"
    assert review["checks"]["has_result"] is False
