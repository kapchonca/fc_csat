from __future__ import annotations

from src.config import load_configs


def test_valid_loading_of_configs() -> None:
    configs = load_configs("configs")
    assert configs["case_templates"]["version"] == "0.1-product-support-debug"
    assert any(tool["id"] == "create_payment_inquiry" for tool in configs["tool_catalog"])
    assert "hard_precondition_edges" in configs["action_graph"]
    assert configs["generation_config"]["variants_per_case"] == 1
