import copy
import importlib.util
import json
import socket
from pathlib import Path

import pytest

from app.analysis.numeric_output_review import review_numeric_output


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Network prohibited')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


def state(metric='gmv', baseline=100, current=120, unit='BRL'):
    return {
        'evidence_bundles': [{'evidence_id': 'ev', 'content': {'result': {
            'status': 'ok', 'parsed': {'metric_name': metric, 'authorized_metric': {'metric_id': metric}},
            'compare_result': {'status': 'ok', 'metric_name': metric, 'baseline_value': baseline,
                               'current_value': current, 'unit': unit,
                               'baseline_date': '2024-05-03', 'current_date': '2024-05-04'},
        }}}],
        'claims': [{'value': f'总变化量为 {current-baseline}', 'source_evidence_ids': ['ev']}],
        'answer': f'总变化量为 {current-baseline}',
        'analysis_presentation': {'changes': [{'metric_name': metric, 'baseline_value': baseline,
            'current_value': current, 'delta_value': current-baseline,
            'delta_ratio': None if baseline == 0 else (current-baseline)*100/baseline,
            'unit': unit, 'change_unit': 'percentage_points' if unit == '%' else unit,
            'delta_ratio_unit': '%', 'evidence_ids': ['ev']}]},
    }


@pytest.mark.parametrize('metric,b,c,unit', [('gmv',100,120,'BRL'),('order_count',120,100,'orders'),
    ('avg_order_value',10,20,'BRL/order'),('freight_rate',13.3,12.8,'%'),('gmv',0,20,'BRL')])
def test_correct_metric_comparisons_and_zero_baseline(metric,b,c,unit):
    assert review_numeric_output(state(metric,b,c,unit))['status'] == 'approved'


@pytest.mark.parametrize('field,value', [('delta_value',2),('baseline_value',120),('current_value',100),
    ('delta_ratio',16.67),('unit','orders'),('baseline_period','2024-05-04')])
def test_structured_errors_rejected(field,value):
    s=state();s['analysis_presentation']['changes'][0][field]=value
    assert review_numeric_output(s)['status']=='rejected'


def test_comparison_card_accepts_full_periods_from_cited_evidence():
    s = state()
    comparison = s['evidence_bundles'][0]['content']['result']['compare_result']
    comparison.update({
        'baseline_date': '2018-07-01',
        'right_end_date': '2018-07-31',
        'current_date': '2018-08-01',
        'left_end_date': '2018-08-31',
    })
    card = s['analysis_presentation']['changes'][0]
    card.update({
        'baseline_date': '2018-07-01',
        'current_date': '2018-08-01',
        'baseline_period': '2018-07-01 至 2018-07-31',
        'current_period': '2018-08-01 至 2018-08-31',
    })

    assert review_numeric_output(s)['status'] == 'approved'

    card['current_period'] = '2018-08-01 至 2018-08-30'
    result = review_numeric_output(s)
    assert result['status'] == 'rejected'
    assert 'changes[0].current_period' in result['issues'][0]


def test_percent_and_percentage_points_are_distinct():
    s=state('freight_rate',13.3,12.8,'%')
    s['analysis_presentation']['changes'][0]['change_unit']='%'
    assert review_numeric_output(s)['status']=='rejected'


def test_correct_card_does_not_hide_bad_text_and_rounding_is_allowed():
    s=state();s['answer']='总变化量为 2 BRL'
    assert review_numeric_output(s)['status']=='rejected'
    s=state('gmv',3,4);s['analysis_presentation']['changes'][0]['delta_ratio']=33.33
    assert review_numeric_output(s)['status']=='approved'
    s['claims'][0]['value']='总变化量为 10'
    assert review_numeric_output(s)['status']=='rejected'


def test_unrelated_evidence_cannot_mask_error():
    s=state();other=copy.deepcopy(s['evidence_bundles'][0]);other['evidence_id']='unrelated'
    other['content']['result']['compare_result']['current_value']=102
    s['evidence_bundles'].append(other)
    s['answer']='总变化量为 2'
    assert review_numeric_output(s)['status']=='rejected'


def test_output_metric_and_direction_cannot_bypass_check():
    s=state();s['analysis_presentation']['changes'][0]['metric_name']='order_count'
    assert review_numeric_output(s)['status']=='rejected'
    s=state();s['answer']='GMV 从100 BRL上升了2 BRL。'
    assert review_numeric_output(s)['status']=='rejected'
    s['answer']='GMV 从100 BRL上升了20 BRL。'
    assert review_numeric_output(s)['status']=='approved'


def test_missing_data_and_no_comparison_are_not_fabricated():
    s=state();s['evidence_bundles'][0]['content']['result']['status']='not_found'
    assert review_numeric_output(s)['status']=='not_applicable'


def test_component_share_is_not_absolute_total_change():
    s=state();s['answer']='总变化 20 BRL。A贡献10 BRL（占总变化 50%），B贡献10 BRL（占 总变化 50%）。'
    assert review_numeric_output(s)['status']=='approved'
    s['answer']='总变化 2 BRL。A贡献10 BRL（占总变化 50%）。'
    assert review_numeric_output(s)['status']=='rejected'


def test_saved_dc05_g_share_text_is_not_false_positive():
    p=Path('docs/testing/resume_bg_ab_dev_20261016/results_v1/traces/f57ccbc5-ecee-5b15-9a2a-548f2eb2bace.run_record.json')
    if not p.exists():
        pytest.skip('Optional newly collected real-run artifact absent')
    saved=json.loads(p.read_text(encoding='utf-8'))['trace']['state_snapshot']
    assert review_numeric_output(saved)['status']=='approved'


def test_real_wrong_answer_and_control_through_critic(tmp_path):
    path=Path('docs/testing/resume_bg_timeout_thinking_diag_20261016/check_numeric_guard_offline.py')
    if not path.exists():
        pytest.skip('Optional local real-response replay artifacts absent')
    spec=importlib.util.spec_from_file_location('numeric_guard_saved_replay',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.BASE=tmp_path
    m.main()
    result=json.loads((tmp_path/'numeric_guard_offline_result_20261017.json').read_text(encoding='utf-8'))
    wrong=result['reviews']['original_wrong_delta'];correct=result['reviews']['corrected_delta_control']
    assert wrong['verdict']=='needs_revision'
    assert wrong['revision_target']=='generate_answer'
    assert wrong['presentation_status']=='partial'
    assert wrong['result']['numeric_review']['status']=='rejected'
    assert wrong['result']['data_review']['status']=='approved'
    assert wrong['result']['claim_review']['status']=='approved'
    assert correct['verdict']=='approved'
    assert correct['result']['numeric_review']['status']=='approved'


@pytest.mark.parametrize('delta,verdict', [(-76.73,'needs_revision'),(-761.27,'approved')])
def test_public_critic_routes_numeric_error_without_llm(delta,verdict):
    from app.graph.nodes.critic import critic
    inner={'status':'ok','parsed':{'metric_name':'freight_value','query_type':'metric_contribution',
                                 'authorized_metric':{'metric_id':'freight_value'}},
           'contribution_result':{'status':'ok','metric_name':'freight_value','source_table':'raw.order_items',
               'baseline_total':5667.01,'current_total':4905.74,'total_delta':-761.27,
               'baseline_items':[],'current_items':[]}}
    wrapped={'result':inner,'request_id':'r','evidence_id':'ev','tool_name':'compare_breakdown_by_dimension','server':'s'}
    s={'task_type':'metric_query','query_results':[wrapped], 'tool_result':wrapped,
       'metric_query_batch_plan':{'plans':[inner['parsed']]},
       'tool_calls':[{'request_id':'r','planned_metric_id':'freight_value','planned_operation':'metric_contribution',
                      'tool_name':wrapped['tool_name'],'server':'s','status':'ok'}],
       'evidence_bundles':[{'evidence_id':'ev','source_ref':wrapped['tool_name'],'content':{'request_id':'r',
                           'metric_id':'freight_value','query_type':'metric_contribution','result':inner}}],
       'claims':[{'value':f'总变化量为 {delta} BRL','source_evidence_ids':['ev']}],
       'answer':f'总变化量为 {delta} BRL',
       'analysis_presentation':{'status':'completed','changes':[{'metric_name':'freight_value',
           'baseline_value':5667.01,'current_value':4905.74,'delta_value':delta,'unit':'BRL','evidence_ids':['ev']}]}}
    result=critic(s,llm_review_mode='critic_llm_review_off')
    assert result['critic_verdict']==verdict
    assert result['critic_result']['data_review']['status']=='approved'
    if verdict=='needs_revision':
        assert result['critic_revision_target']=='generate_answer'
        assert result['analysis_presentation']['status']=='partial'
