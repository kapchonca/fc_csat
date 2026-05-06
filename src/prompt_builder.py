from __future__ import annotations

import json
from typing import Any


CONDITION_RENDERING_INSTRUCTIONS = {
    "correct": "The assistant completes the customer request smoothly and safely.",
    "skip_step": (
        "The assistant moves ahead before giving the customer the key answer they "
        "needed. The customer reacts to the missing answer in ordinary language, "
        "then the assistant checks the missing detail and completes the request."
    ),
    "extra_step": (
        "The assistant spends time on a broader account or payment review that the "
        "customer did not ask for and that is not needed for this request. The task "
        "is still completed."
    ),
    "wrong_order": (
        "The assistant starts an inquiry before first showing whether the payment "
        "actually needs one. The customer reacts to still not having the answer they "
        "asked for, then the assistant checks the payment and completes the request."
    ),
    "wrong_tool": (
        "The assistant gives a broad or adjacent account answer instead of answering "
        "the specific payment question. The customer reacts because their question is "
        "still unanswered, then the assistant checks the specific payment and completes "
        "the request."
    ),
    "wrong_parameter": (
        "The assistant initially relies on a vague or mistaken detail and risks focusing "
        "on the wrong payment. The customer provides a clarifying detail naturally, then "
        "the assistant uses the corrected detail and completes the request."
    ),
}


def build_prompt(
    case_spec: dict[str, Any],
    tool_catalog: list[dict[str, Any]],
    generation_config: dict[str, Any],
) -> str:
    return build_dialogue_prompt(
        case_spec=case_spec,
        tool_catalog=tool_catalog,
        generation_config=generation_config,
        dialogue_plan=None,
    )


def build_dialogue_plan_prompt(
    case_spec: dict[str, Any],
    tool_catalog: list[dict[str, Any]],
    generation_config: dict[str, Any],
) -> str:
    tools = {tool["id"]: tool for tool in tool_catalog}
    relevant_tools = _relevant_tools(case_spec, tools)
    plan_config = _dialogue_plan_config(generation_config)
    message_count = plan_config["message_count"]
    role_pattern = plan_config["role_pattern"]
    forbidden_tool_ids = ", ".join(sorted(tool["id"] for tool in relevant_tools))

    return "\n".join(
        [
            "Prompt type: dialogue_plan",
            "You are planning a controlled retail banking support dialogue for a survey experiment.",
            "Create only a high-level message plan. Do not write the final dialogue yet.",
            "",
            "Hidden experiment metadata. Use it only to plan the dialogue; never reveal it.",
            f"Case id: {case_spec['case_id']}",
            f"User goal: {_humanize_task(case_spec['task'])}",
            f"Condition: {case_spec['condition']}",
            f"Expected outcome: {case_spec['expected_outcome']}",
            f"Hidden action trace: {json.dumps(case_spec['trace'])}",
            f"Error metadata: {json.dumps(case_spec.get('error'))}",
            f"Experimental labels: {json.dumps(case_spec['labels'])}",
            "",
            "Private banking support context. Use this for grounding, not for wording.",
            _format_relevant_tools(relevant_tools),
            "",
            "Planning requirements:",
            f"- Return exactly {message_count} plan items.",
            f"- Use this exact role pattern: {json.dumps(role_pattern)}.",
            "- Each item must describe the purpose of that message in natural language.",
            "- The plan must make the customer-visible behavior clear without naming the technical mistake.",
            "- The plan must make the expected final outcome clear.",
            "- Refer to banking actions only in customer-facing plain language.",
            "- The customer should sound like a normal banking customer, not an evaluator or system designer.",
            "- The customer should react to what is missing, confusing, or unhelpful, not explain internal process.",
            "",
            "Condition-specific planning instruction:",
            _condition_planning_instruction(case_spec, tools),
            "",
            "Forbidden behavior:",
            "- Do not mention internal tool ids, trace, case id, condition names, labels, or error labels.",
            "- Do not make the customer name the mistake directly.",
            "- Avoid evaluator phrases such as you skipped, wrong order, wrong tool, required check, correct sequence, or operations history is not the same.",
            "- Avoid process-heavy words such as retrieve, lookup, backend, workflow, or sequence.",
            f"- Do not mention exact action ids: {forbidden_tool_ids}.",
            "- Do not add markdown, comments, explanations, headings, or text outside the JSON.",
            "",
            "Output format:",
            "Return only a valid JSON array in this exact shape:",
            '[{"role":"user","purpose":"..."},{"role":"assistant","purpose":"..."}]',
        ]
    )


def build_dialogue_prompt(
    case_spec: dict[str, Any],
    tool_catalog: list[dict[str, Any]],
    generation_config: dict[str, Any],
    dialogue_plan: list[dict[str, Any]] | None = None,
) -> str:
    tools = {tool["id"]: tool for tool in tool_catalog}
    relevant_tools = _relevant_tools(case_spec, tools)
    forbidden_tool_ids = ", ".join(sorted(tool["id"] for tool in relevant_tools))
    semantic_variant = _semantic_variant(generation_config)
    if dialogue_plan is None:
        message_requirement = (
            f"- Write approximately {generation_config['output_message_min']} to "
            f"{generation_config['output_message_max']} total messages."
        )
        plan_lines = []
    else:
        message_requirement = f"- Write exactly {len(dialogue_plan)} total messages."
        plan_lines = [
            "",
            "Validated dialogue plan. Follow it exactly for roles and message purposes:",
            json.dumps(dialogue_plan, ensure_ascii=False, indent=2),
        ]

    return "\n".join(
        [
            "Prompt type: dialogue_render",
            "You are a natural-language renderer for a controlled survey experiment.",
            "Setting: retail banking support chat between one customer and one assistant.",
            "The interaction structure is predefined. Do not invent new labels, outcomes, or hidden actions.",
            "",
            "Hidden experiment metadata. Use it only to render the dialogue; never reveal it.",
            f"Case id: {case_spec['case_id']}",
            f"User goal: {_humanize_task(case_spec['task'])}",
            f"Condition: {case_spec['condition']}",
            f"Expected outcome: {case_spec['expected_outcome']}",
            f"Hidden action trace: {json.dumps(case_spec['trace'])}",
            f"Error metadata: {json.dumps(case_spec.get('error'))}",
            f"Experimental labels: {json.dumps(case_spec['labels'])}",
            f"Semantic variant id: {semantic_variant['id']}",
            *plan_lines,
            "",
            "Private banking support context. Use this for grounding, not for wording.",
            _format_relevant_tools(relevant_tools),
            "",
            "How the condition should be visible:",
            CONDITION_RENDERING_INSTRUCTIONS[case_spec["condition"]],
            "",
            "Dialogue requirements:",
            message_requirement,
            "- Follow the role order from the validated plan when a plan is provided.",
            "- Use realistic but fictional banking details such as merchant names, amounts, dates, and account endings.",
            "- Make the expected final outcome clear in the final assistant message.",
            "- The assistant can describe customer-facing checks, delays, or issues in plain language.",
            "- The customer must not sound like they know the assistant's internal action graph or implementation.",
            "- The customer should not explicitly diagnose the technical mistake; they should react to the unsatisfied request.",
            "- The assistant should not narrate backend steps or use process-heavy wording.",
            "- Preserve the semantic style from the validated plan.",
            f"- Semantic rendering instruction: {semantic_variant['instruction']}",
            "",
            "Forbidden behavior:",
            "- Do not mention internal tool ids, trace, case id, condition names, labels, or error labels.",
            "- Do not mention the semantic variant id or say that a style variant is being used.",
            "- Do not use words such as trace, tool, condition, wrong_parameter, missing_input, skip_step, wrong_order, wrong_tool, task_completed, or task_failed.",
            "- Avoid evaluator or pipeline phrases such as you skipped, wrong order, wrong tool, required check, correct sequence, retrieve, lookup, or operations history is not the same.",
            f"- Do not mention exact action ids: {forbidden_tool_ids}.",
            "- Do not add markdown, comments, explanations, headings, or text outside the JSON.",
            "",
            "Output format:",
            "Return only a valid JSON array in this exact shape:",
            '[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]',
        ]
    )


def _relevant_tools(
    case_spec: dict[str, Any],
    tools: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    action_ids: list[str] = []
    for action in case_spec["trace"]:
        action_id = action.split("@", 1)[-1]
        if action_id in tools and action_id not in action_ids:
            action_ids.append(action_id)

    error = case_spec.get("error")
    if error and error.get("replacement") in tools:
        replacement = error["replacement"]
        if replacement not in action_ids:
            action_ids.append(replacement)
    return [tools[action_id] for action_id in action_ids]


def _format_relevant_tools(relevant_tools: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {tool['description']}"
        for tool in relevant_tools
    )


def _condition_planning_instruction(
    case_spec: dict[str, Any],
    tools: dict[str, dict[str, Any]],
) -> str:
    condition = case_spec["condition"]
    error = case_spec.get("error") or {}
    if condition == "correct":
        return (
            "Plan a smooth exchange where the customer asks about a payment, the assistant "
            "answers the payment question, and the assistant completes the requested inquiry."
        )
    if condition == "skip_step":
        return (
            "Plan an exchange where the assistant starts moving toward a support inquiry "
            "before telling the customer what is happening with the payment. The customer "
            "should ask a normal follow-up like whether the payment went through or what "
            "its status is. The assistant then answers that missing question and completes "
            "the inquiry."
        )
    if condition == "extra_step":
        return (
            "Plan an exchange where the assistant has enough detail to answer the specific "
            "payment question but first spends a turn on a broad, unnecessary review such "
            "as recent payments or other products. The customer did not ask for that broad "
            "review and should not be the one requesting it. The assistant still completes "
            "the original request."
        )
    if condition == "wrong_order":
        return (
            "Plan an exchange where the assistant opens or starts an inquiry before first "
            "telling the customer whether the payment appears to need one. The customer "
            "should not say this is the wrong order; they should ask why they still do not "
            "know what happened with the payment. The assistant then checks the payment, "
            "uses that answer to update the inquiry, and completes the request."
        )
    if condition == "wrong_tool":
        return (
            "Plan an exchange where the assistant gives a broad adjacent answer, such as "
            "a list or general payment activity, instead of answering the customer's question "
            "about one specific payment. The customer should say that this still does not "
            "tell them what is happening with that payment. The assistant then checks the "
            "specific payment and completes the request."
        )
    if condition == "wrong_parameter":
        return (
            "Plan an exchange where the assistant initially relies on a vague or mistaken "
            "detail, such as only an amount, the wrong date, or the wrong merchant, and "
            "starts to focus on a payment that may not be the customer's payment. The "
            "customer should clarify naturally, without saying parameter error. The assistant "
            "then uses the corrected detail and completes the request."
        )
    return CONDITION_RENDERING_INSTRUCTIONS[condition]


def _action_description(
    action_id: str | None,
    tools: dict[str, dict[str, Any]],
) -> str:
    if action_id and action_id in tools:
        return tools[action_id]["description"]
    if action_id:
        return action_id.replace("_", " ")
    return "the required banking support step"


def _dialogue_plan_config(generation_config: dict[str, Any]) -> dict[str, Any]:
    return generation_config["dialogue_plan"]


def _semantic_variant(generation_config: dict[str, Any]) -> dict[str, str]:
    variant = generation_config.get("semantic_variant")
    if isinstance(variant, dict):
        return {
            "id": str(variant.get("id", "default")),
            "instruction": str(
                variant.get(
                    "instruction",
                    "Use neutral wording without changing the scenario.",
                )
            ),
        }
    return {
        "id": "default",
        "instruction": "Use neutral wording without changing the scenario.",
    }


def _humanize_task(task: str) -> str:
    if task == "cancel_payment":
        return "cancel a pending payment before it is processed"
    return task.replace("_", " ")
