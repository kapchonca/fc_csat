from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from src.case_spec_generator import generate_case_specs, save_case_specs
from src.config import load_configs, read_json
from src.parser import parse_dialogue, parse_dialogue_plan
from src.prompt_builder import (
    build_dialogue_plan_prompt,
    build_dialogue_prompt,
    build_prompt,
)
from src.renderers import get_renderer
from src.validator import validate_dialogue, validate_dialogue_plan

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress display
    tqdm = None


DEFAULT_OUTPUT_DIR = Path("outputs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled synthetic banking support dialogue generator."
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing tool_catalog, action_graph, case_templates, and generation_config.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where generated files will be written.",
    )
    parser.add_argument(
        "--generation-config",
        default=None,
        help="Optional path to a generation config JSON file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser(
        "generate-case-specs",
        help="Generate case_specs.json.",
    )
    _add_common_subcommand_options(generate_parser)

    render_parser = subparsers.add_parser(
        "render-dialogues",
        help="Render dialogues from an existing case_specs.json.",
    )
    _add_common_subcommand_options(render_parser)
    render_parser.add_argument(
        "--case-specs",
        default=None,
        help="Path to case_specs.json. Defaults to output-dir/case_specs.json.",
    )

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Generate case specs and render dialogues.",
    )
    _add_common_subcommand_options(run_all_parser)

    args = parser.parse_args()
    if args.command == "generate-case-specs":
        command_generate_case_specs(args)
    elif args.command == "render-dialogues":
        command_render_dialogues(args)
    elif args.command == "run-all":
        command_run_all(args)


def _add_common_subcommand_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help="Directory containing input configuration files.",
    )
    parser.add_argument(
        "--output-dir",
        default=argparse.SUPPRESS,
        help="Directory where generated files will be written.",
    )
    parser.add_argument(
        "--generation-config",
        default=argparse.SUPPRESS,
        help="Optional path to a generation config JSON file.",
    )
    parser.add_argument(
        "--limit-dialogues",
        type=int,
        default=argparse.SUPPRESS,
        help="Optional maximum number of dialogue variants to render.",
    )


def command_generate_case_specs(args: argparse.Namespace) -> None:
    configs = load_configs(args.config_dir, args.generation_config)
    output_dir = _timestamped_output_dir(Path(args.output_dir))
    specs = generate_case_specs(
        configs["tool_catalog"],
        configs["action_graph"],
        configs["case_templates"],
    )
    case_specs_path = output_dir / "case_specs.json"
    save_case_specs(specs, case_specs_path)
    print(f"Generated {len(specs)} case specs at {case_specs_path}")


def command_render_dialogues(args: argparse.Namespace) -> None:
    configs = load_configs(args.config_dir, args.generation_config)
    requested_output_dir = Path(args.output_dir)
    case_specs_path = (
        Path(args.case_specs)
        if args.case_specs
        else requested_output_dir / "case_specs.json"
    )
    output_dir = _timestamped_output_dir(requested_output_dir)
    case_specs = read_json(case_specs_path)
    _render_dialogues(configs, case_specs, output_dir, getattr(args, "limit_dialogues", None))


def command_run_all(args: argparse.Namespace) -> None:
    configs = load_configs(args.config_dir, args.generation_config)
    output_dir = _timestamped_output_dir(Path(args.output_dir))
    case_specs = generate_case_specs(
        configs["tool_catalog"],
        configs["action_graph"],
        configs["case_templates"],
    )
    save_case_specs(case_specs, output_dir / "case_specs.json")
    _render_dialogues(configs, case_specs, output_dir, getattr(args, "limit_dialogues", None))


def _render_dialogues(
    configs: dict[str, Any],
    case_specs: list[dict[str, Any]],
    output_dir: Path,
    limit_dialogues: int | None = None,
) -> None:
    if limit_dialogues is not None and limit_dialogues < 1:
        raise ValueError("--limit-dialogues must be a positive integer.")

    output_dir.mkdir(parents=True, exist_ok=True)
    generation_config = configs["generation_config"]
    selected_specs = _select_case_specs(case_specs, generation_config.get("include_cases", []))
    variants_per_case = generation_config["variants_per_case"]
    semantic_variants = _semantic_variants(generation_config)
    max_parallel_requests = _max_parallel_requests(generation_config)

    dialogues: list[dict[str, Any]] = []
    debug_records: list[dict[str, Any]] = []
    usage_summary = _new_usage_summary(generation_config)
    failure_reasons: Counter[str] = Counter(
        {
            "invalid_json": 0,
            "wrong_role": 0,
            "empty_message": 0,
            "empty_plan_purpose": 0,
            "message_count_mismatch": 0,
            "role_pattern_mismatch": 0,
            "role_plan_mismatch": 0,
            "render_exception": 0,
        }
    )

    tasks = _build_render_tasks(
        selected_specs=selected_specs,
        semantic_variants=semantic_variants,
        variants_per_case=variants_per_case,
        limit_dialogues=limit_dialogues,
    )
    debug_jsonl_path = output_dir / "debug.jsonl"
    dialogues_jsonl_path = output_dir / "dialogues.jsonl"
    results: list[dict[str, Any]] = []
    task_by_future: dict[Any, dict[str, Any]] = {}
    with (
        debug_jsonl_path.open("w", encoding="utf-8") as debug_handle,
        dialogues_jsonl_path.open("w", encoding="utf-8") as dialogues_handle,
        ThreadPoolExecutor(max_workers=max_parallel_requests) as executor,
    ):
        futures = [
            executor.submit(_render_dialogue_task, configs, task)
            for task in tasks
        ]
        task_by_future = {future: task for future, task in zip(futures, tasks)}
        for future in _progress_futures(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = _failed_task_result(task_by_future[future], exc)
            results.append(result)
            _write_jsonl_record(debug_handle, result["debug_record"])
            if result["dialogue"] is not None:
                _write_jsonl_record(dialogues_handle, result["dialogue"])
            debug_handle.flush()
            dialogues_handle.flush()

    for result in sorted(results, key=lambda item: item["sequence"]):
        if result["dialogue"] is not None:
            dialogues.append(result["dialogue"])
        else:
            for error in result["validation_errors"]:
                failure_reasons[error] += 1
        debug_records.append(result["debug_record"])
        for usage_record in result["usage_records"]:
            _add_usage(
                usage_summary["total"],
                usage_record["usage"],
                usage_record["cost"],
            )
            _add_usage(
                usage_summary["by_stage"][usage_record["stage"]],
                usage_record["usage"],
                usage_record["cost"],
            )

    generated = len(tasks)
    summary = {
        "dataset_version": configs["case_templates"]["version"],
        "generated": generated,
        "passed": len(dialogues),
        "failed": generated - len(dialogues),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "model": generation_config["model"],
        "temperature": generation_config.get("temperature"),
        "seed": generation_config.get("seed"),
        "semantic_variants": [variant["id"] for variant in semantic_variants],
        "max_parallel_requests": max_parallel_requests,
        "usage": _finalize_usage_summary(usage_summary),
    }

    _write_json(output_dir / "dialogues.json", dialogues)
    _write_json(output_dir / "run_summary.json", summary)
    print(
        f"Rendered {generated} variants: {summary['passed']} passed, "
        f"{summary['failed']} failed. Outputs written to {output_dir}"
    )


def _select_case_specs(
    case_specs: list[dict[str, Any]],
    include_cases: list[str],
) -> list[dict[str, Any]]:
    if not include_cases:
        return case_specs

    by_id = {spec["case_id"]: spec for spec in case_specs}
    missing = [case_id for case_id in include_cases if case_id not in by_id]
    if missing:
        raise ValueError(f"include_cases references unknown case ids: {missing}")
    return [by_id[case_id] for case_id in include_cases]


def _progress_futures(futures: list[Any]) -> Any:
    completed = as_completed(futures)
    if tqdm is None:
        return completed
    return tqdm(
        completed,
        total=len(futures),
        desc="Rendering dialogues",
        unit="dialogue",
    )


def _dialogue_plan_enabled(generation_config: dict[str, Any]) -> bool:
    plan_config = generation_config.get("dialogue_plan")
    return isinstance(plan_config, dict) and plan_config.get("enabled", False)


def _build_render_tasks(
    selected_specs: list[dict[str, Any]],
    semantic_variants: list[dict[str, str]],
    variants_per_case: int,
    limit_dialogues: int | None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    sequence = 0
    for case_spec in selected_specs:
        for semantic_variant in semantic_variants:
            for variant_id in range(1, variants_per_case + 1):
                if limit_dialogues is not None and len(tasks) >= limit_dialogues:
                    return tasks
                tasks.append(
                    {
                        "sequence": sequence,
                        "case_spec": case_spec,
                        "semantic_variant": semantic_variant,
                        "variant_id": variant_id,
                    }
                )
                sequence += 1
    return tasks


def _render_dialogue_task(
    configs: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    generation_config = configs["generation_config"]
    case_spec = task["case_spec"]
    semantic_variant = task["semantic_variant"]
    variant_id = task["variant_id"]
    renderer = get_renderer(generation_config)
    render_config = dict(generation_config)
    render_config["variant_id"] = variant_id
    render_config["case_id"] = case_spec["case_id"]
    render_config["semantic_variant"] = semantic_variant

    plan_prompt = None
    raw_plan_output = None
    plan_parse_errors: list[str] = []
    parsed_plan = None
    dialogue_plan = None
    plan_validation = None
    plan_usage = None
    plan_cost = None
    dialogue_usage = None
    dialogue_cost = None
    usage_records: list[dict[str, Any]] = []

    if _dialogue_plan_enabled(generation_config):
        plan_config = dict(render_config)
        plan_config["render_stage"] = "plan"
        plan_prompt = build_dialogue_plan_prompt(
            case_spec,
            configs["tool_catalog"],
            plan_config,
        )
        raw_plan_output = renderer.render(plan_prompt, plan_config)
        plan_usage, plan_cost = _capture_render_usage(renderer, generation_config)
        usage_records.append({"stage": "plan", "usage": plan_usage, "cost": plan_cost})
        plan_parse_result = parse_dialogue_plan(raw_plan_output)
        plan_parse_errors = plan_parse_result.errors
        parsed_plan = plan_parse_result.items
        plan_validation = validate_dialogue_plan(
            case_spec=case_spec,
            plan=parsed_plan,
            generation_config=generation_config,
            tool_catalog=configs["tool_catalog"],
            variant_id=variant_id,
            parse_errors=plan_parse_result.errors,
        )
        if plan_validation.status == "passed":
            dialogue_plan = parsed_plan
            render_config["render_stage"] = "dialogue"
            prompt = build_dialogue_prompt(
                case_spec,
                configs["tool_catalog"],
                render_config,
                dialogue_plan=dialogue_plan,
            )
            raw_output = renderer.render(prompt, render_config)
            dialogue_usage, dialogue_cost = _capture_render_usage(
                renderer,
                generation_config,
            )
            usage_records.append(
                {"stage": "dialogue", "usage": dialogue_usage, "cost": dialogue_cost}
            )
            parse_result = parse_dialogue(raw_output)
            validation = validate_dialogue(
                case_spec=case_spec,
                messages=parse_result.messages,
                generation_config=generation_config,
                tool_catalog=configs["tool_catalog"],
                variant_id=variant_id,
                parse_errors=parse_result.errors,
                dialogue_plan=dialogue_plan,
            )
        else:
            prompt = None
            raw_output = None
            parse_result = None
            validation = plan_validation
    else:
        prompt = build_prompt(case_spec, configs["tool_catalog"], render_config)
        raw_output = renderer.render(prompt, render_config)
        dialogue_usage, dialogue_cost = _capture_render_usage(renderer, generation_config)
        usage_records.append(
            {"stage": "dialogue", "usage": dialogue_usage, "cost": dialogue_cost}
        )
        parse_result = parse_dialogue(raw_output)
        validation = validate_dialogue(
            case_spec=case_spec,
            messages=parse_result.messages,
            generation_config=generation_config,
            tool_catalog=configs["tool_catalog"],
            variant_id=variant_id,
            parse_errors=parse_result.errors,
        )

    parsed_messages = parse_result.messages if parse_result is not None else None
    parser_errors = parse_result.errors if parse_result is not None else []
    dialogue = None
    if validation.status == "passed" and parsed_messages is not None:
        dialogue = {
            "dialogue_id": _dialogue_id(
                case_spec["case_id"],
                semantic_variant["id"],
                variant_id,
            ),
            "case_id": case_spec["case_id"],
            "task": case_spec["task"],
            "condition": case_spec["condition"],
            "semantic_variant": semantic_variant["id"],
            "variant_id": variant_id,
            "expected_outcome": case_spec["expected_outcome"],
            "labels": case_spec["labels"],
            "messages": parsed_messages,
        }

    debug_record = {
        "case_id": case_spec["case_id"],
        "semantic_variant": semantic_variant["id"],
        "variant_id": variant_id,
        "plan_prompt": plan_prompt,
        "raw_plan_output": raw_plan_output,
        "plan_usage": plan_usage,
        "plan_cost": plan_cost,
        "parsed_plan": parsed_plan,
        "plan_parser_errors": plan_parse_errors,
        "plan_validator_status": plan_validation.status
        if plan_validation is not None
        else None,
        "plan_validator_errors": plan_validation.errors
        if plan_validation is not None
        else [],
        "prompt": prompt,
        "raw_output": raw_output,
        "dialogue_usage": dialogue_usage,
        "dialogue_cost": dialogue_cost,
        "parsed_output": parsed_messages,
        "parser_errors": parser_errors,
        "validator_status": validation.status,
        "validator_errors": validation.errors,
        "validator_warnings": validation.warnings,
    }

    return {
        "sequence": task["sequence"],
        "dialogue": dialogue,
        "debug_record": debug_record,
        "validation_errors": validation.errors,
        "usage_records": usage_records,
    }


def _failed_task_result(task: dict[str, Any], exc: Exception) -> dict[str, Any]:
    case_spec = task["case_spec"]
    semantic_variant = task["semantic_variant"]
    variant_id = task["variant_id"]
    error_code = "render_exception"
    debug_record = {
        "case_id": case_spec["case_id"],
        "semantic_variant": semantic_variant["id"],
        "variant_id": variant_id,
        "plan_prompt": None,
        "raw_plan_output": None,
        "plan_usage": None,
        "plan_cost": None,
        "parsed_plan": None,
        "plan_parser_errors": [],
        "plan_validator_status": None,
        "plan_validator_errors": [],
        "prompt": None,
        "raw_output": None,
        "dialogue_usage": None,
        "dialogue_cost": None,
        "parsed_output": None,
        "parser_errors": [],
        "validator_status": "failed",
        "validator_errors": [error_code],
        "validator_warnings": [],
        "exception_type": type(exc).__name__,
        "exception": str(exc),
    }
    return {
        "sequence": task["sequence"],
        "dialogue": None,
        "debug_record": debug_record,
        "validation_errors": [error_code],
        "usage_records": [],
    }


def _semantic_variants(generation_config: dict[str, Any]) -> list[dict[str, str]]:
    variants = generation_config.get("semantic_variants", [])
    if not variants:
        return [
            {
                "id": "default",
                "instruction": "Use neutral wording without changing the scenario.",
            }
        ]
    return [
        {
            "id": variant["id"],
            "instruction": variant["instruction"],
        }
        for variant in variants
    ]


def _dialogue_id(case_id: str, semantic_variant_id: str, variant_id: int) -> str:
    if semantic_variant_id == "default":
        return f"{case_id}_v{variant_id:02d}"
    return f"{case_id}__{semantic_variant_id}_v{variant_id:02d}"


def _max_parallel_requests(generation_config: dict[str, Any]) -> int:
    value = generation_config.get("max_parallel_requests", 1)
    return value if isinstance(value, int) and value > 0 else 1


def _timestamped_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return base_dir / timestamp


def _new_usage_summary(generation_config: dict[str, Any]) -> dict[str, Any]:
    pricing = generation_config.get("pricing", {})
    currency = pricing.get("currency", "USD") if isinstance(pricing, dict) else "USD"
    return {
        "currency": currency,
        "pricing": pricing if isinstance(pricing, dict) else {},
        "total": _empty_usage_bucket(),
        "by_stage": {
            "plan": _empty_usage_bucket(),
            "dialogue": _empty_usage_bucket(),
        },
    }


def _empty_usage_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "billable_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "provider_reported_cost": 0.0,
    }


def _record_render_usage(
    usage_summary: dict[str, Any],
    renderer: Any,
    generation_config: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = getattr(renderer, "last_response_metadata", {}) or {}
    usage = metadata.get("usage", {}) if isinstance(metadata, dict) else {}
    normalized_usage = _normalize_usage(usage)
    cost = _calculate_cost(normalized_usage, generation_config)

    _add_usage(usage_summary["total"], normalized_usage, cost)
    _add_usage(usage_summary["by_stage"][stage], normalized_usage, cost)
    return normalized_usage, cost


def _capture_render_usage(
    renderer: Any,
    generation_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = getattr(renderer, "last_response_metadata", {}) or {}
    usage = metadata.get("usage", {}) if isinstance(metadata, dict) else {}
    normalized_usage = _normalize_usage(usage)
    cost = _calculate_cost(normalized_usage, generation_config)
    return normalized_usage, cost


def _normalize_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = _int_usage(usage.get("prompt_tokens"))
    completion_tokens = _int_usage(usage.get("completion_tokens"))
    total_tokens = _int_usage(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    prompt_details = usage.get("prompt_tokens_details", {})
    cached_prompt_tokens = 0
    if isinstance(prompt_details, dict):
        cached_prompt_tokens = _int_usage(prompt_details.get("cached_tokens"))
    billable_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
    provider_reported_cost = usage.get("cost")
    return {
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "billable_prompt_tokens": billable_prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "provider_reported_cost": round(float(provider_reported_cost), 8)
        if isinstance(provider_reported_cost, (int, float))
        else None,
    }


def _int_usage(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _calculate_cost(
    usage: dict[str, Any],
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    pricing = generation_config.get("pricing", {})
    currency = "USD"
    if not isinstance(pricing, dict):
        pricing = {}
    else:
        currency = pricing.get("currency", "USD")

    input_rate = _float_price(pricing.get("input_per_1m_tokens"))
    cached_input_rate = _float_price(
        pricing.get("cached_input_per_1m_tokens"),
        default=input_rate,
    )
    output_rate = _float_price(pricing.get("output_per_1m_tokens"))
    amount = (
        usage["billable_prompt_tokens"] * input_rate
        + usage["cached_prompt_tokens"] * cached_input_rate
        + usage["completion_tokens"] * output_rate
    ) / 1_000_000
    return {
        "currency": currency,
        "amount": round(amount, 8),
        "pricing_available": bool(pricing),
        "provider_reported_amount": usage["provider_reported_cost"],
    }


def _float_price(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and value >= 0:
        return float(value)
    return default


def _add_usage(bucket: dict[str, Any], usage: dict[str, Any], cost: dict[str, Any]) -> None:
    bucket["calls"] += 1
    for key in (
        "prompt_tokens",
        "cached_prompt_tokens",
        "billable_prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        bucket[key] += usage[key]
    bucket["cost"] = round(bucket["cost"] + cost["amount"], 8)
    if usage["provider_reported_cost"] is not None:
        bucket["provider_reported_cost"] = round(
            bucket["provider_reported_cost"] + usage["provider_reported_cost"],
            8,
        )


def _finalize_usage_summary(usage_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "currency": usage_summary["currency"],
        "pricing": usage_summary["pricing"],
        "total": usage_summary["total"],
        "by_stage": usage_summary["by_stage"],
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            _write_jsonl_record(handle, record)


def _write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
