from __future__ import annotations

from src.case_spec_generator import generate_case_specs
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
    spec = specs["cancel_payment_correct"]
    assert spec["trace"] == [
        "verify_identity",
        "find_payment",
        "check_payment_status",
        "cancel_payment",
        "done",
    ]
    assert spec["expected_outcome"] == "task_completed"
    assert spec["labels"] == ["correct", "task_completed"]


def test_skip_step_generation() -> None:
    _, specs = _specs()
    spec = specs["cancel_payment_skip_step_check_payment_status"]
    assert "check_payment_status" not in spec["trace"]
    assert spec["trace"] == ["verify_identity", "find_payment", "cancel_payment", "done"]
    assert spec["expected_outcome"] == "task_failed"


def test_wrong_order_generation_from_hard_precondition_edges() -> None:
    configs, specs = _specs()
    spec = specs["cancel_payment_wrong_order_verify_identity_find_payment"]
    assert ["verify_identity", "find_payment"] in configs["action_graph"]["hard_precondition_edges"]
    assert spec["trace"].index("find_payment") < spec["trace"].index("verify_identity")
    assert spec["expected_outcome"] == "task_failed"


def test_wrong_tool_generation_from_confusion_edges() -> None:
    configs, specs = _specs()
    spec = specs["cancel_payment_wrong_tool_find_payment"]
    assert ["find_payment", "list_recent_transactions"] in configs["action_graph"]["confusion_edges"]
    assert "find_payment" not in spec["trace"]
    assert "list_recent_transactions" in spec["trace"]
    assert spec["error"]["replacement"] == "list_recent_transactions"


def test_removed_scenarios_are_not_generated() -> None:
    _, specs = _specs()
    assert all("recovered" not in case_id for case_id in specs)
    assert all("not_recovered" not in case_id for case_id in specs)
    assert all("tool_failure" not in case_id for case_id in specs)


def test_wrong_parameter_generation() -> None:
    _, specs = _specs()
    spec = specs["cancel_payment_wrong_parameter_find_payment"]
    assert spec["trace"] == [
        "verify_identity",
        "wrong_parameter@find_payment",
        "done",
    ]
    assert spec["expected_outcome"] == "task_failed"
    assert spec["labels"] == ["wrong_parameter", "task_failed"]
