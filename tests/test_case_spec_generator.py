from __future__ import annotations

from src.case_spec_generator import CaseSpecError, generate_case_specs
from src.config import load_configs


def _specs():
    configs = load_configs("configs")
    specs = generate_case_specs(
        configs["tool_catalog"],
        configs["action_graph"],
        configs["case_templates"],
    )
    return configs, {spec["case_id"]: spec for spec in specs}


def test_case_id_uniqueness() -> None:
    configs = load_configs("configs")
    specs = generate_case_specs(
        configs["tool_catalog"],
        configs["action_graph"],
        configs["case_templates"],
    )
    ids = [spec["case_id"] for spec in specs]
    assert len(ids) == len(set(ids))


def test_correct_case_generation() -> None:
    _, specs = _specs()
    spec = specs["payment_inquiry_correct"]
    assert spec["trace"] == [
        "verify_customer_session",
        "get_status_payment",
        "create_payment_inquiry",
        "done",
    ]
    assert spec["expected_outcome"] == "task_completed"
    assert spec["labels"] == ["correct", "task_completed"]


def test_skip_step_generation() -> None:
    _, specs = _specs()
    spec = specs["payment_inquiry_skip_step_get_status_payment"]
    assert spec["trace"] == [
        "verify_customer_session",
        "skip_step@get_status_payment",
        "get_status_payment",
        "create_payment_inquiry",
        "done",
    ]
    assert spec["expected_outcome"] == "task_completed"
    assert spec["labels"] == ["skip_step", "recovered", "task_completed"]


def test_skip_step_not_recovered_generation() -> None:
    _, specs = _specs()
    spec = specs["payment_inquiry_skip_step_not_recovered_get_status_payment"]
    assert spec["trace"] == [
        "verify_customer_session",
        "skip_step@get_status_payment",
        "create_payment_inquiry",
        "done",
    ]
    assert spec["expected_outcome"] == "task_failed"
    assert spec["labels"] == ["skip_step", "not_recovered", "task_failed"]


def test_extra_step_generation_from_extra_step_candidates() -> None:
    configs, specs = _specs()
    spec = specs["payment_inquiry_extra_step_get_operation_details"]
    assert ["get_status_payment", "get_operation_details"] in configs["action_graph"][
        "extra_step_candidates"
    ]
    assert spec["trace"] == [
        "verify_customer_session",
        "get_status_payment",
        "get_operation_details",
        "create_payment_inquiry",
        "done",
    ]
    assert spec["error"] == {
        "type": "extra_step",
        "at": "get_operation_details",
        "anchor": "get_status_payment",
    }
    assert spec["expected_outcome"] == "task_completed"


def test_extra_step_generation_requires_extra_step_candidates() -> None:
    configs = load_configs("configs")
    action_graph = dict(configs["action_graph"])
    action_graph["extra_step_candidates"] = []

    try:
        generate_case_specs(
            configs["tool_catalog"],
            action_graph,
            configs["case_templates"],
        )
    except CaseSpecError as exc:
        assert "extra_step_candidates" in str(exc)
    else:
        raise AssertionError("Expected CaseSpecError for missing extra_step_candidates.")


def test_wrong_order_generation_from_hard_precondition_edges() -> None:
    configs, specs = _specs()
    spec = specs["payment_inquiry_wrong_order_get_status_payment_create_payment_inquiry"]
    assert ["get_status_payment", "create_payment_inquiry"] in configs["action_graph"][
        "hard_precondition_edges"
    ]
    assert spec["trace"].index("create_payment_inquiry") < spec["trace"].index(
        "get_status_payment"
    )
    assert spec["trace"].count("create_payment_inquiry") == 2
    assert spec["expected_outcome"] == "task_completed"
    assert spec["labels"] == ["wrong_order", "recovered", "task_completed"]


def test_wrong_order_not_recovered_generation() -> None:
    _, specs = _specs()
    spec = specs[
        "payment_inquiry_wrong_order_not_recovered_get_status_payment_create_payment_inquiry"
    ]
    assert spec["trace"] == [
        "verify_customer_session",
        "create_payment_inquiry",
        "get_status_payment",
        "done",
    ]
    assert spec["expected_outcome"] == "task_failed"
    assert spec["labels"] == ["wrong_order", "not_recovered", "task_failed"]


def test_wrong_tool_generation_from_confusion_edges() -> None:
    configs, specs = _specs()
    spec = specs["payment_inquiry_wrong_tool_get_status_payment"]
    assert ["get_status_payment", "get_operations_payment"] in configs["action_graph"][
        "confusion_edges"
    ]
    assert spec["trace"] == [
        "verify_customer_session",
        "get_operations_payment",
        "get_status_payment",
        "create_payment_inquiry",
        "done",
    ]
    assert spec["error"]["replacement"] == "get_operations_payment"
    assert spec["expected_outcome"] == "task_completed"
    assert spec["labels"] == ["wrong_tool", "recovered", "task_completed"]


def test_wrong_tool_not_recovered_generation() -> None:
    _, specs = _specs()
    spec = specs["payment_inquiry_wrong_tool_not_recovered_get_status_payment"]
    assert spec["trace"] == [
        "verify_customer_session",
        "get_operations_payment",
        "create_payment_inquiry",
        "done",
    ]
    assert spec["expected_outcome"] == "task_failed"
    assert spec["labels"] == ["wrong_tool", "not_recovered", "task_failed"]


def test_wrong_parameter_generation() -> None:
    _, specs = _specs()
    spec = specs["payment_inquiry_wrong_parameter_get_status_payment"]
    assert spec["trace"] == [
        "verify_customer_session",
        "wrong_parameter@get_status_payment",
        "get_status_payment",
        "create_payment_inquiry",
        "done",
    ]
    assert spec["expected_outcome"] == "task_completed"
    assert spec["labels"] == ["wrong_parameter", "recovered", "task_completed"]


def test_wrong_parameter_not_recovered_generation() -> None:
    _, specs = _specs()
    spec = specs["payment_inquiry_wrong_parameter_not_recovered_get_status_payment"]
    assert spec["trace"] == [
        "verify_customer_session",
        "wrong_parameter@get_status_payment",
        "done",
    ]
    assert spec["expected_outcome"] == "task_failed"
    assert spec["labels"] == ["wrong_parameter", "not_recovered", "task_failed"]


def test_tool_failure_scenarios_are_not_generated() -> None:
    _, specs = _specs()
    assert all("tool_failure" not in case_id for case_id in specs)
