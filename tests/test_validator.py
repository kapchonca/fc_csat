from __future__ import annotations

from src.case_spec_generator import generate_case_specs
from src.config import load_configs
from src.validator import validate_dialogue


def _case(case_id: str):
    configs = load_configs("configs/product_support_debug")
    specs = generate_case_specs(
        configs["tool_catalog"],
        configs["action_graph"],
        configs["case_templates"],
    )
    by_id = {spec["case_id"]: spec for spec in specs}
    return configs, by_id[case_id]


def test_validator_does_not_check_internal_label_leaks() -> None:
    configs, spec = _case("payment_inquiry_wrong_parameter_get_status_payment")
    messages = [
        {"role": "user", "content": "Please help cancel this payment."},
        {"role": "assistant", "content": "I can check that."},
        {"role": "user", "content": "Thanks."},
        {"role": "assistant", "content": "This is a wrong_parameter case. Done."},
    ]
    result = validate_dialogue(
        spec,
        messages,
        configs["generation_config"],
        configs["tool_catalog"],
    )
    assert result.status == "passed"
    assert "internal_label_leak" not in result.errors


def test_validator_rejects_invalid_roles() -> None:
    configs, spec = _case("payment_inquiry_correct")
    messages = [
        {"role": "user", "content": "Cancel a payment."},
        {"role": "agent", "content": "The cancellation is confirmed."},
    ]
    result = validate_dialogue(
        spec,
        messages,
        configs["generation_config"],
        configs["tool_catalog"],
    )
    assert result.status == "failed"
    assert "wrong_role" in result.errors


def test_validator_does_not_check_outcome_wording() -> None:
    configs, spec = _case("payment_inquiry_correct")
    messages = [
        {"role": "user", "content": "Cancel a payment."},
        {"role": "assistant", "content": "I can look into that."},
        {"role": "user", "content": "Thanks."},
        {"role": "assistant", "content": "I am unable to complete that cancellation."},
    ]
    result = validate_dialogue(
        spec,
        messages,
        configs["generation_config"],
        configs["tool_catalog"],
    )
    assert result.status == "passed"
    assert "wrong_outcome" not in result.errors
