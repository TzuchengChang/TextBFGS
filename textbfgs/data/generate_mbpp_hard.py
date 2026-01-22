#!/usr/bin/env python3
"""
Generate a "hard" subset of MBPP tasks.

Definition (hard):
  A task is considered hard if a single initial model generation does NOT pass
  all base tests OR does NOT pass all plus tests (EvalPlus).

This script mirrors `generate_humaneval_hard.py` but targets the MBPP benchmark.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import check_correctness, get_groundtruth
from evalplus.eval import PASS

from textbfgs.code.client import QwenClient


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def build_initial_prompt(problem: Dict, extra_inst: str) -> str:
    prompt = problem["prompt"].rstrip()
    if not prompt.endswith("\n"):
        prompt += "\n"
    suffix = (
        "\nPlease output complete, executable Python code. "
        "Keep the original function signature and return type. "
        "Do not print explanations or additional text."
    )
    if extra_inst:
        suffix += f"\n{extra_inst}"
    return f"{prompt}{suffix}"


def strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    if len(parts) >= 3:
        body = parts[1]
        if "\n" in body and body.split("\n", 1)[0].strip().isalpha():
            body = body.split("\n", 1)[1]
        return body.strip()
    return text.strip()


def strip_improved_tags(text: str) -> str:
    return (
        text.replace("<IMPROVED>", "")
        .replace("</IMPROVED>", "")
        .strip()
    )


def clean_code(text: str) -> str:
    return strip_improved_tags(strip_code_fences(text))


def parse_task_ids(raw: str, all_ids: Iterable[str]) -> List[str]:
    if not raw or raw.lower() == "all":
        return list(all_ids)
    return [tid.strip() for tid in raw.split(",") if tid.strip()]


def eval_code(task_id: str, code: str, problem: Dict, problem_hash) -> Tuple[float, Dict[str, Any]]:
    expected = get_groundtruth({task_id: problem}, problem_hash, [])
    try:
        result = check_correctness(
            dataset="mbpp",
            completion_id=0,
            problem=problem,
            solution=code,
            expected_output=expected[task_id],
            base_only=False,
            fast_check=True,
            identifier=task_id,
        )
    except Exception as exc:  # noqa: BLE001
        return 0.0, {
            "task_id": task_id,
            "error": f"evaluation error: {exc}",
            "base_all_passed": False,
            "plus_all_passed": False,
            "base_passed": 0,
            "base_total": 0,
            "plus_passed": 0,
            "plus_total": 0,
        }

    base_status, base_details = result["base"]
    plus_status, plus_details = result["plus"]

    def summarize(status, details):
        try:
            if details is not None and len(details) > 0:
                arr = list(details) if hasattr(details, "__iter__") else []
                passed = sum(1 for x in arr if x)
                total = len(arr)
                all_ok = status == PASS and passed == total
            else:
                passed, total = 0, 0
                all_ok = status == PASS
        except Exception:
            passed, total = 0, 0
            all_ok = status == PASS
        return passed, total, all_ok

    base_passed, base_total, base_ok = summarize(base_status, base_details)
    plus_passed, plus_total, plus_ok = summarize(plus_status, plus_details)
    combined = 0.5 * ((base_passed / base_total if base_total else 0.0) + (plus_passed / plus_total if plus_total else 0.0))

    return combined, {
        "task_id": task_id,
        "base_all_passed": base_ok,
        "plus_all_passed": plus_ok,
        "base_passed": base_passed,
        "base_total": base_total,
        "plus_passed": plus_passed,
        "plus_total": plus_total,
    }


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Generate MBPP-Hard (initial generation fails base or plus).")
    parser.add_argument("--model", default="Qwen3-235B-A22B", help="model name")
    parser.add_argument("--api-key", default="Bearer YOUR_API_KEY_HERE", help="API key")
    parser.add_argument("--base-url", default="https://api.example.com/v1", help="API base URL")
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="top_p")
    parser.add_argument("--max-tokens", type=int, default=4096, help="max generated tokens")
    parser.add_argument("--request-timeout", type=int, default=300, help="request timeout (seconds)")
    parser.add_argument("--max-retries", type=int, default=3, help="request retries")
    parser.add_argument("--task-ids", type=str, default="all", help="comma-separated task ids or 'all'")
    parser.add_argument("--max-tasks", type=int, default=0, help="limit number of tasks (0 = no limit)")
    parser.add_argument("--generation-hint", type=str, default="", help="extra hint appended to prompt")
    parser.add_argument(
        "--output",
        type=str,
        default="textbfgs/data/mbpp-hard/mbpp-hard.jsonl",
        help="output JSONL path",
    )
    args = parser.parse_args()

    problems = get_mbpp_plus()
    selected = parse_task_ids(args.task_ids, problems.keys())
    if args.max_tasks and args.max_tasks > 0:
        selected = selected[: args.max_tasks]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = QwenClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    problem_hash = get_mbpp_plus_hash()

    hard_count = 0
    total = len(selected)
    logging.info("Scanning %d MBPP tasks for hard subset...", total)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, tid in enumerate(selected, start=1):
            problem = problems[tid]
            prompt = build_initial_prompt(problem, args.generation_hint)

            try:
                raw = client.generate_message(
                    messages=[
                        {"role": "system", "content": "You are a senior Python engineer."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                raw = f"__GENERATION_FAILED__: {exc}"

            code = clean_code(raw)
            _, metrics = eval_code(tid, code, problem, problem_hash)

            is_hard = (not metrics.get("base_all_passed", False)) or (not metrics.get("plus_all_passed", False))
            if is_hard:
                hard_count += 1
                record = {
                    "task_id": tid,
                    "entry_point": problem.get("entry_point"),
                    "prompt": problem.get("prompt"),
                    "initial_solution": code,
                    "metrics": metrics,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if idx % 10 == 0 or idx == total:
                logging.info("Progress %d/%d | hard=%d", idx, total, hard_count)

            time.sleep(0.0)

    logging.info("Done. Hard tasks: %d/%d. Output: %s", hard_count, total, str(out_path))


if __name__ == "__main__":
    main()

