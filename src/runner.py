from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.case_spec_generator import generate_case_specs, save_case_specs
from src.config import load_configs, read_json
from src.parser import parse_dialogue
from src.prompt_builder import build_prompt
from src.renderers import get_renderer
from src.validator import validate_dialogue


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


def command_generate_case_specs(args: argparse.Namespace) -> None:
    configs = load_configs(args.config_dir)
    output_dir = Path(args.output_dir)
    specs = generate_case_specs(
        configs["tool_catalog"],
        configs["action_graph"],
        configs["case_templates"],
    )
    case_specs_path = output_dir / "case_specs.json"
    save_case_specs(specs, case_specs_path)
    print(f"Generated {len(specs)} case specs at {case_specs_path}")


def command_render_dialogues(args: argparse.Namespace) -> None:
    configs = load_configs(args.config_dir)
    output_dir = Path(args.output_dir)
    case_specs_path = Path(args.case_specs) if args.case_specs else output_dir / "case_specs.json"
    case_specs = read_json(case_specs_path)
    _render_dialogues(configs, case_specs, output_dir)


def command_run_all(args: argparse.Namespace) -> None:
    configs = load_configs(args.config_dir)
    output_dir = Path(args.output_dir)
    case_specs = generate_case_specs(
        configs["tool_catalog"],
        configs["action_graph"],
        configs["case_templates"],
    )
    save_case_specs(case_specs, output_dir / "case_specs.json")
    _render_dialogues(configs, case_specs, output_dir)


def _render_dialogues(
    configs: dict[str, Any],
    case_specs: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_config = configs["generation_config"]
    selected_specs = _select_case_specs(case_specs, generation_config.get("include_cases", []))
    renderer = get_renderer(generation_config)
    variants_per_case = generation_config["variants_per_case"]

    dialogues: list[dict[str, Any]] = []
    debug_records: list[dict[str, Any]] = []
    failure_reasons: Counter[str] = Counter(
        {
            "invalid_json": 0,
            "wrong_role": 0,
            "empty_message": 0,
            "internal_label_leak": 0,
            "wrong_outcome": 0,
        }
    )

    for case_spec in selected_specs:
        for variant_id in range(1, variants_per_case + 1):
            render_config = dict(generation_config)
            render_config["variant_id"] = variant_id
            render_config["case_id"] = case_spec["case_id"]
            prompt = build_prompt(case_spec, configs["tool_catalog"], render_config)
            raw_output = renderer.render(prompt, render_config)
            parse_result = parse_dialogue(raw_output)
            validation = validate_dialogue(
                case_spec=case_spec,
                messages=parse_result.messages,
                generation_config=generation_config,
                tool_catalog=configs["tool_catalog"],
                variant_id=variant_id,
                parse_errors=parse_result.errors,
            )

            if validation.status == "passed" and parse_result.messages is not None:
                dialogues.append(
                    {
                        "dialogue_id": f"{case_spec['case_id']}_v{variant_id:02d}",
                        "case_id": case_spec["case_id"],
                        "task": case_spec["task"],
                        "condition": case_spec["condition"],
                        "variant_id": variant_id,
                        "expected_outcome": case_spec["expected_outcome"],
                        "labels": case_spec["labels"],
                        "messages": parse_result.messages,
                    }
                )
            else:
                for error in validation.errors:
                    failure_reasons[error] += 1

            debug_records.append(
                {
                    "case_id": case_spec["case_id"],
                    "variant_id": variant_id,
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "parsed_output": parse_result.messages,
                    "parser_errors": parse_result.errors,
                    "validator_status": validation.status,
                    "validator_errors": validation.errors,
                    "validator_warnings": validation.warnings,
                }
            )

    generated = len(selected_specs) * variants_per_case
    summary = {
        "dataset_version": configs["case_templates"]["version"],
        "generated": generated,
        "passed": len(dialogues),
        "failed": generated - len(dialogues),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "model": generation_config["model"],
        "temperature": generation_config.get("temperature"),
        "seed": generation_config.get("seed"),
    }

    _write_json(output_dir / "dialogues.json", dialogues)
    _write_jsonl(output_dir / "debug.jsonl", debug_records)
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


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
