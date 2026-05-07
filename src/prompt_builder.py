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
    "skip_step_not_recovered": (
        "The assistant moves ahead before giving the customer the key answer they "
        "needed. The customer reacts to the missing answer in ordinary language, "
        "but the assistant does not resolve the gap and the request remains incomplete."
    ),
    "extra_step": (
        "The assistant spends time on an additional nearby support action that the "
        "customer did not ask for and that is not needed for this request. The task is "
        "still completed."
    ),
    "wrong_order": (
        "The assistant performs a downstream action before the key customer-facing "
        "answer is established. The customer reacts to still not having the answer they "
        "asked for, then the assistant corrects course and completes the request."
    ),
    "wrong_order_not_recovered": (
        "The assistant performs a downstream action before the key customer-facing "
        "answer is established. The customer reacts to still not having the answer they "
        "asked for, and the assistant does not complete the request correctly."
    ),
    "wrong_tool": (
        "The assistant gives concrete adjacent information instead of the specific "
        "customer-facing answer the user asked for. The customer reacts because their "
        "question is still unanswered, then the assistant corrects course and completes "
        "the request."
    ),
    "wrong_tool_not_recovered": (
        "The assistant gives concrete adjacent information instead of the specific "
        "customer-facing answer the user asked for. The customer reacts because their "
        "question is still unanswered, but the assistant does not correct course and "
        "the request remains incomplete."
    ),
    "wrong_parameter": (
        "The assistant initially relies on a vague or mistaken detail and risks focusing "
        "on the wrong customer object or record. The customer provides a clarifying "
        "detail naturally, then the assistant uses the corrected detail and completes "
        "the request."
    ),
    "wrong_parameter_not_recovered": (
        "The assistant initially relies on a vague or mistaken detail and risks focusing "
        "on the wrong customer object or record. The customer provides a clarifying "
        "detail naturally, but the assistant does not use it properly and the request "
        "remains incomplete."
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
    output_format = _output_format_instruction(
        generation_config,
        wrapper_key="plan",
        text_property="purpose",
    )

    return "\n".join(
        [
            "Prompt type: dialogue_plan",
            "You are planning a controlled retail banking support dialogue for a survey experiment.",
            "Create only a high-level message plan. Do not write the final dialogue yet.",
            "",
            "Hidden experiment metadata. Use it only to plan the dialogue; never reveal it.",
            f"Case id: {case_spec['case_id']}",
            f"User goal: {_user_goal(case_spec)}",
            f"Condition: {case_spec['condition']}",
            f"Expected outcome: {case_spec['expected_outcome']}",
            f"Hidden action trace: {json.dumps(case_spec['trace'])}",
            f"Error metadata: {json.dumps(case_spec.get('error'))}",
            f"Experimental labels: {json.dumps(case_spec['labels'])}",
            f"Key customer-facing action: {_action_description(case_spec.get('target_action'), tools)}",
            f"Downstream or follow-up action: {_action_description(case_spec.get('downstream_action'), tools)}",
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
            "- Do not frame non-recovery as missing system access, system outage, or inability to check from the chat.",
            "- For unresolved cases, the assistant had enough customer information but failed to use the right customer-facing action.",
            "",
            "Condition-specific planning instruction:",
            _condition_planning_instruction(case_spec, tools),
            "",
            "Forbidden behavior:",
            "- Do not mention internal tool ids, trace, case id, condition names, labels, or error labels.",
            "- Do not make the customer name the mistake directly.",
            "- Avoid evaluator phrases such as you skipped, wrong order, wrong tool, required check, correct sequence, or the adjacent information is not the same.",
            "- Avoid process-heavy words such as retrieve, lookup, backend, workflow, or sequence.",
            f"- Do not mention exact action ids: {forbidden_tool_ids}.",
            "- Do not add markdown, comments, explanations, headings, or text outside the JSON.",
            "",
            "Output format:",
            output_format,
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
    output_format = _output_format_instruction(
        generation_config,
        wrapper_key="messages",
        text_property="content",
    )
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
            f"User goal: {_user_goal(case_spec)}",
            f"Condition: {case_spec['condition']}",
            f"Expected outcome: {case_spec['expected_outcome']}",
            f"Hidden action trace: {json.dumps(case_spec['trace'])}",
            f"Error metadata: {json.dumps(case_spec.get('error'))}",
            f"Experimental labels: {json.dumps(case_spec['labels'])}",
            f"Key customer-facing action: {_action_description(case_spec.get('target_action'), tools)}",
            f"Downstream or follow-up action: {_action_description(case_spec.get('downstream_action'), tools)}",
            f"Semantic variant id: {semantic_variant['id']}",
            *plan_lines,
            "",
            "Private banking support context. Use this for grounding, not for wording.",
            _format_relevant_tools(relevant_tools),
            "",
            "How the condition should be visible:",
            _condition_rendering_instruction(case_spec, tools),
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
            "- Do not frame non-recovery as missing system access, system outage, or inability to check from the chat.",
            "- If the request is unresolved, make the unresolved part be the user's key answer, even if a case number or follow-up has been created.",
            "- Preserve the semantic style from the validated plan.",
            f"- Semantic rendering instruction: {semantic_variant['instruction']}",
            "",
            "Forbidden behavior:",
            "- Do not mention internal tool ids, trace, case id, condition names, labels, or error labels.",
            "- Do not mention the semantic variant id or say that a style variant is being used.",
            "- Do not use words such as trace, tool, condition, wrong_parameter, missing_input, skip_step, wrong_order, wrong_tool, task_completed, or task_failed.",
            "- Avoid evaluator or pipeline phrases such as you skipped, wrong order, wrong tool, required check, correct sequence, retrieve, lookup, or the adjacent information is not the same.",
            f"- Do not mention exact action ids: {forbidden_tool_ids}.",
            "- Do not add markdown, comments, explanations, headings, or text outside the JSON.",
            "",
            "Output format:",
            output_format,
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
    if error and error.get("at") in tools:
        action_id = error["at"]
        if action_id not in action_ids:
            action_ids.append(action_id)
    if error and error.get("replacement") in tools:
        replacement = error["replacement"]
        if replacement not in action_ids:
            action_ids.append(replacement)
    return [tools[action_id] for action_id in action_ids]


def _format_relevant_tools(relevant_tools: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {_plain_action_description(tool['description'])}"
        for tool in relevant_tools
    )


def _condition_rendering_instruction(
    case_spec: dict[str, Any],
    tools: dict[str, dict[str, Any]],
) -> str:
    condition = case_spec["condition"]
    if condition not in CONDITION_RENDERING_INSTRUCTIONS:
        return (
            "The assistant's customer-visible behavior must follow the case metadata "
            "and leave the expected outcome clear."
        )

    context = _support_context(case_spec, tools)
    instruction = CONDITION_RENDERING_INSTRUCTIONS[condition]
    if condition.endswith("_not_recovered"):
        return (
            f"{instruction} The unresolved issue should be that the customer still does "
            f"not have the key answer: {context['target_description']}. A downstream "
            f"or follow-up action may be taken: {context['downstream_description']}. "
            "Do not present this as a system limitation."
        )
    return instruction


def _condition_planning_instruction(
    case_spec: dict[str, Any],
    tools: dict[str, dict[str, Any]],
) -> str:
    condition = case_spec["condition"]
    error = case_spec.get("error") or {}
    context = _support_context(case_spec, tools)
    if condition == "correct":
        return (
            f"Plan a smooth exchange where the customer asks for this goal: "
            f"{context['user_goal']}. The assistant provides the key customer-facing "
            f"answer ({context['target_description']}) and completes any appropriate "
            "follow-up."
        )
    if condition == "skip_step":
        return (
            f"Plan an exchange where the assistant moves toward the downstream or "
            f"follow-up action ({context['downstream_description']}) before providing "
            f"the key answer ({context['target_description']}). The customer should "
            "react in ordinary language to the missing answer. The assistant then gives "
            "the missing answer and completes the request."
        )
    if condition == "skip_step_not_recovered":
        return (
            f"Plan an exchange where the assistant moves toward the downstream or "
            f"follow-up action ({context['downstream_description']}) before providing "
            f"the key answer ({context['target_description']}). The customer should "
            "ask for that missing answer in ordinary language. The assistant should "
            "not provide the key answer; the final assistant message may give a case "
            "number or follow-up path, but the customer's original question remains "
            "unanswered. Do not blame missing system access."
        )
    if condition == "extra_step":
        extra_description = _action_description(error.get("at"), tools)
        return (
            f"Plan an exchange where the assistant has enough detail to satisfy the "
            f"user goal ({context['user_goal']}) but first spends a turn on an "
            f"unnecessary nearby action: {extra_description}. The customer did not ask "
            "for that extra action and should not be the one requesting it. The assistant "
            "still completes the original request."
        )
    if condition == "wrong_order":
        return (
            f"Plan an exchange where the assistant performs the downstream or follow-up "
            f"action ({context['downstream_description']}) before first establishing "
            f"the key answer ({context['target_description']}). The customer should "
            "not name the ordering issue; they should react to still missing the answer. "
            "The assistant then provides the key answer and completes the request."
        )
    if condition == "wrong_order_not_recovered":
        return (
            f"Plan an exchange where the assistant performs the downstream or follow-up "
            f"action ({context['downstream_description']}) before first establishing "
            f"the key answer ({context['target_description']}). The customer should "
            "not name the ordering issue; they should react to still missing the answer. "
            "The assistant may attempt to address the key action too late, but does not "
            "turn it into a clear answer for the customer. The final assistant message "
            "may confirm follow-up exists, but the key answer remains unresolved."
        )
    if condition == "wrong_tool":
        replacement_description = _action_description(error.get("replacement"), tools)
        return (
            f"Plan an exchange where the assistant gives concrete adjacent information "
            f"({replacement_description}) instead of the key answer "
            f"({context['target_description']}). The adjacent information should look "
            "plausibly related, not like a generic FAQ. The customer should react that "
            "their actual question is still unanswered. The assistant then corrects "
            "course and completes the request."
        )
    if condition == "wrong_tool_not_recovered":
        replacement_description = _action_description(error.get("replacement"), tools)
        return (
            f"Plan an exchange where the assistant gives concrete adjacent information "
            f"({replacement_description}) instead of the key answer "
            f"({context['target_description']}). The adjacent information should be "
            "specific and plausible, not a generic explanation. The customer should "
            "react that their actual question is still unanswered. The assistant should "
            "not correct course; the final assistant message may create or mention "
            "follow-up, but the key answer remains unresolved."
        )
    if condition == "wrong_parameter":
        return (
            f"Plan an exchange where the assistant initially relies on vague, incomplete, "
            f"or misread customer details while trying to perform the key action "
            f"({context['target_description']}). The assistant focuses on a similar but "
            "wrong customer object or record. The customer clarifies naturally, without "
            "saying parameter error. The assistant then uses the corrected detail and "
            "completes the request."
        )
    if condition == "wrong_parameter_not_recovered":
        return (
            f"Plan an exchange where the assistant initially relies on vague, incomplete, "
            f"or misread customer details while trying to perform the key action "
            f"({context['target_description']}). The assistant focuses on a similar but "
            "wrong customer object or record. The customer clarifies naturally, without "
            "saying parameter error. The assistant should keep relying on the wrong object "
            "or use the downstream follow-up based on that wrong object. The final assistant "
            "message may offer follow-up, but the key answer remains unresolved."
        )
    return CONDITION_RENDERING_INSTRUCTIONS[condition]


def _action_description(
    action_id: str | None,
    tools: dict[str, dict[str, Any]],
) -> str:
    if action_id and action_id in tools:
        return _plain_action_description(tools[action_id]["description"])
    if action_id:
        return action_id.replace("_", " ")
    return "the required banking support step"


def _plain_action_description(description: str) -> str:
    replacements = {
        "Retrieve ": "Review ",
        "Generate ": "Prepare ",
    }
    for prefix, replacement in replacements.items():
        if description.startswith(prefix):
            return replacement + description[len(prefix):]
    return description


def _support_context(
    case_spec: dict[str, Any],
    tools: dict[str, dict[str, Any]],
) -> dict[str, str]:
    return {
        "user_goal": _user_goal(case_spec),
        "target_description": _action_description(case_spec.get("target_action"), tools),
        "downstream_description": _action_description(
            case_spec.get("downstream_action"),
            tools,
        ),
    }


def _output_format_instruction(
    generation_config: dict[str, Any],
    wrapper_key: str,
    text_property: str,
) -> str:
    item_shape = (
        f'{{"role":"user","{text_property}":"..."}},'
        f'{{"role":"assistant","{text_property}":"..."}}'
    )
    if generation_config.get("response_format") == "json_schema":
        return (
            "Return only a valid JSON object in this exact shape: "
            f'{{"{wrapper_key}":[{item_shape}]}}'
        )
    return (
        "Return only a valid JSON array in this exact shape: "
        f"[{item_shape}]"
    )


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


def _user_goal(case_spec: dict[str, Any]) -> str:
    user_goal = case_spec.get("user_goal")
    if isinstance(user_goal, str) and user_goal.strip():
        return user_goal.strip()
    return _humanize_task(case_spec["task"])


def _humanize_task(task: str) -> str:
    if task == "cancel_payment":
        return "cancel a pending payment before it is processed"
    return task.replace("_", " ")
