from evaluation.claim_verification.build_pilot_annotation_draft import build


def test_mutations_stay_construction_derived_and_route_conservatively() -> None:
    source = [{"case_id": "C1", "original_answer_seen_by_critic": "value is 1"}]
    real = [{"case_id": "C1", "atomic_claims": [{"claim_id": "R1"}], "requirement_coverage": {}}]
    pilot = [
        {"pilot_id": "REAL-C1", "source_case_id": "C1", "provenance": "real_trace_draft", "answer": "value is 1"},
        {"pilot_id": "NUM-C1", "source_case_id": "C1", "provenance": "targeted_mutation", "mutation_type": "numeric_fact_error", "original_text": "1", "changed_text": "2", "answer": "value is 2", "task_requirements": []},
    ]
    # The production builder requires 30 rows; this unit test checks the row
    # semantics by extending to 30 unique numeric mutations.
    pilot.extend({"pilot_id": f"NUM-{i}", "source_case_id": "C1", "provenance": "targeted_mutation", "mutation_type": "numeric_fact_error", "original_text": "1", "changed_text": "2", "answer": "value is 2", "task_requirements": []} for i in range(2, 30))
    output = build(pilot, real, source)
    numeric = output[1]
    assert numeric["label_origin"] == "construction_derived_mutation_draft"
    assert numeric["atomic_claims"][0]["claim_type"] == "numeric_fact"
    assert numeric["atomic_claims"][0]["expected_route"] == "deterministic"
    assert numeric["human_verified"] is False
