import copy
import json
import socket
from pathlib import Path

import pytest

from evaluation.resume_bg_protocol import score_bg_run, score_bg_run_v4


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Offline scorer must not access network or database')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


def example():
    case={'case_id':'D','protocol_task_type':'dimension_statistical_contribution', 'target_metric':'freight_value',
          'dimension':'product_category','scope':'platform','comparison_period':{'baseline':'2024-05-03','current':'2024-05-04'},
          'expected_result':{'baseline_metric':10,'current_metric':12,'delta':2,'expected_contributions':{'A':1,'B':1}},
          'unit':'BRL','output_units':{'metric_unit':'BRL','change_unit':'BRL','component_unit':'BRL'},'tolerance':{'absolute':0.03}}
    card={'metric_name':'freight_value','dimension':'product_category','baseline_date':'2024-05-03','current_date':'2024-05-04',
          'baseline_value':10,'current_value':12,'total_delta':2,'unit':'BRL',
          'items':[{'dimension_value':'A','delta_value':1},{'dimension_value':'B','delta_value':1}],'evidence_ids':['e']}
    trace={'state_snapshot':{'scope_type':'platform','answer':'Total change is 2 BRL; A and B each contribute 1 BRL, descriptive only.',
        'terminal_code':'ENOUGH_EVIDENCE','analysis_presentation':{'status':'completed','contributions':[card]},
        'claims':[{'value':'Change 2 BRL','source_evidence_ids':['e']}],
        'tool_calls':[{'request_id':'r','call_status':'succeeded'}],
        'evidence_bundles':[{'evidence_id':'e','request_id':'r','content':{'request_id':'r','result':{'status':'ok','parsed':{
            'metric_name':'freight_value','authorized_metric':{'metric_id':'freight_value'},'dimension':'product_category',
            'baseline_date':'2024-05-03','current_date':'2024-05-04'}}}}]}}
    return case,trace


def score(case,trace,**kw):
    return score_bg_run_v4(case,arm='basic_tool_loop',trace=trace,lifecycle_status='completed',**kw)


def test_list_adapter_preserves_v3_and_requires_human_review():
    case,trace=example();before=copy.deepcopy((case,trace))
    old=score_bg_run(case,arm='basic_tool_loop',trace=trace,lifecycle_status='completed')
    new=score(case,trace)
    assert old['scorer_version']=='resume-bg-score-v3' and not old['numeric_checks']['baseline']
    assert new['scorer_version']=='resume-bg-score-v4' and all(new['numeric_checks'].values())
    assert new['task_acceptance_status']=='pending_manual_review' and not new['task_acceptance_pass']
    assert score(case,trace,human_review={'decision':'pass'})['task_acceptance_pass']
    assert (case,trace)==before


@pytest.mark.parametrize('field,value', [('current_value',13),('total_delta',3),('unit','orders'),
    ('current_date','2024-05-03'),('metric_name','gmv'),('dimension','customer_state'),('evidence_ids',['wrong'])])
def test_wrong_card_is_not_hidden_by_target_state(field,value):
    case,trace=example();trace['state_snapshot']['analysis_presentation']['contributions'][0][field]=value
    assert not score(case,trace,human_review={'decision':'pass'})['task_acceptance_pass']


def test_duplicates_are_rejected():
    case,trace=example();cards=trace['state_snapshot']['analysis_presentation']['contributions'];cards.append(copy.deepcopy(cards[0]))
    r=score(case,trace,human_review={'decision':'pass'})
    assert r['schema_adaptation']=='missing_or_ambiguous_matching_card' and not r['task_acceptance_pass']


def test_other_net_is_not_exact_group_coverage():
    case,trace=example();card=trace['state_snapshot']['analysis_presentation']['contributions'][0]
    card['items']=[{'dimension_value':'A','delta_value':1},{'dimension_value':'Other','delta_value':1}]
    r=score(case,trace,human_review={'decision':'pass'})
    assert r['numeric_checks']['baseline'] and not r['numeric_checks']['components']
    assert r['dimension_coverage']['displayed_net_values_match']
    assert r['dimension_coverage']['omitted_net_aggregate_matches']
    assert not r['dimension_coverage']['exact_group_coverage'] and not r['task_acceptance_pass']


def test_percentage_level_unit_does_not_supply_percentage_points():
    case,trace=example();case['unit']='%';case['output_units']={'metric_unit':'%','change_unit':'percentage_point','component_unit':'percentage_point'}
    card=trace['state_snapshot']['analysis_presentation']['contributions'][0];card['unit']='%'
    r=score(case,trace);assert r['numeric_checks']['metric_unit'] and not r['numeric_checks']['change_unit']
    card['change_unit']='percentage_points';card['component_unit']='percentage_points'
    assert all(score(case,trace)['numeric_checks'].values())


def test_saved_followup_offline_score_keeps_coverage_failure():
    root=Path('docs/testing/dc05_g_numeric_followup_20261017/traces')
    files=list(root.glob('*.run_record.json'))
    if not files:
        pytest.skip('Optional local real-run artifact absent')
    raw=json.loads(files[0].read_text(encoding='utf-8'))
    case=next(c for c in json.loads(Path('docs/testing/resume_bg_ab_dev_20261016/cases_6.json').read_text(encoding='utf-8'))['cases'] if c['case_id']=='DC05')
    r=score_bg_run_v4(case,arm='governed_workflow',trace=raw['trace'],lifecycle_status='completed')
    assert r['numeric_checks']['baseline'] and r['numeric_checks']['current'] and r['numeric_checks']['delta']
    assert r['numeric_checks']['metric_unit'] and r['numeric_checks']['change_unit'] and r['numeric_checks']['component_unit']
    assert not r['numeric_checks']['components'] and r['dimension_coverage']['omitted_net_aggregate_matches']
    assert not r['task_acceptance_pass'] and r['manual_review_required']
