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
    assert spec["trace"] == [
        "verify_identity",
        "find_payment",
        "skip_step@check_payment_status",
        "check_payment_status",
        "cancel_payment",
        "done",
    ]
    assert spec["expected_outcome"] == "task_completed"
    assert spec["labels"] == ["skip_step", "recovered", "task_completed"]


def test_extra_step_generation_from_extra_step_candidates() -> None:
    configs, specs = _specs()
    spec = specs["cancel_payment_extra_step_list_recent_transactions"]
    assert ["find_payment", "list_recent_transactions"] in configs["action_graph"][
        "extra_step_candidates"
    ]
    assert spec["trace"] == [
        "verify_identity",
        "find_payment",
        "list_recent_transactions",
        "check_payment_status",
        "cancel_payment",
        "done",
    ]
    assert spec["error"] == {
        "type": "extra_step",
        "at": "list_recent_transactions",
        "anchor": "find_payment",
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
    spec = specs["cancel_payment_wrong_order_verify_identity_find_payment"]
    assert ["verify_identity", "find_payment"] in configs["action_graph"]["hard_precondition_edges"]
    assert spec["trace"].index("find_payment") < spec["trace"].index("verify_identity")
    assert spec["trace"].count("verify_identity") == 2
    assert spec["expected_outcome"] == "task_completed"
    assert spec["labels"] == ["wrong_order", "recovered", "task_completed"]


def test_wrong_tool_generation_from_confusion_edges() -> None:
    configs, specs = _specs()
    spec = specs["cancel_payment_wrong_tool_find_payment"]
    assert ["find_payment", "list_recent_transactions"] in configs["action_graph"]["confusion_edges"]
    assert "find_payment" in spec["trace"]
    assert "list_recent_transactions" in spec["trace"]
    assert spec["trace"].index("list_recent_transactions") < spec["trace"].index("find_payment")
    assert spec["error"]["replacement"] == "list_recent_transactions"
    assert spec["expected_outcome"] == "task_completed"
    assert spec["labels"] == ["wrong_tool", "recovered", "task_completed"]


def test_removed_scenarios_are_not_generated() -> None:
    _, specs = _specs()
    assert all("not_recovered" not in case_id for case_id in specs)
    assert all("tool_failure" not in case_id for case_id in specs)


def test_wrong_parameter_generation() -> None:
    _, specs = _specs()
    spec = specs["cancel_payment_wrong_parameter_find_payment"]
    assert spec["trace"] == [
        "verify_identity",
        "wrong_parameter@find_payment",
        "find_payment",
        "check_payment_status",
        "cancel_payment",
        "done",
    ]
    assert spec["expected_outcome"] == "task_completed"
    assert spec["labels"] == ["wrong_parameter", "recovered", "task_completed"]
