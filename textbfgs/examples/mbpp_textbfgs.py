#!/usr/bin/env python3
"""
Run TextBFGS (Textual-BFGS) optimization on MBPP/MBPP+ using TextGrad.

This script mirrors the HumanEval runner but targets the MBPP benchmark.
It is an anonymized version of the internal runner.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import check_correctness, evaluate as evalplus_evaluate, get_groundtruth
from evalplus.eval import PASS

import textgrad as tg
from textgrad.engine import EngineLM
from textgrad.variable import Variable

from textbfgs.code.optimizer import HessianProxyKB, TextualBFGS
from textbfgs.code.client import QwenClient


class TextGradEngine(EngineLM):
    """Adapter between `QwenClient` and TextGrad's `EngineLM`."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        timeout: int,
        max_retries: int,
        system_prompt: str,
        log_callback=None,
        nothink: bool = False,
    ):
        self.model_string = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.nothink = nothink
        self.client = QwenClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.log_callback = log_callback

    def generate(self, prompt, system_prompt=None, **kwargs):
        sys_msg = system_prompt or self.system_prompt
        payload = {
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if kwargs.get("nothink", self.nothink):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload = {k: v for k, v in payload.items() if v is not None}
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]

        call_start_time = time.time()
        call_timestamp = datetime.now().isoformat()

        try:
            response = self.client.generate_message(messages=messages, **payload)
            call_duration = time.time() - call_start_time

            if self.log_callback:
                self.log_callback(
                    {
                        "timestamp": call_timestamp,
                        "duration_seconds": call_duration,
                        "system_prompt": sys_msg,
                        "user_prompt": prompt,
                        "response": response,
                        "temperature": payload.get("temperature"),
                        "top_p": payload.get("top_p"),
                        "max_tokens": payload.get("max_tokens"),
                        "success": True,
                    }
                )

            return response
        except Exception as exc:
            call_duration = time.time() - call_start_time
            if self.log_callback:
                self.log_callback(
                    {
                        "timestamp": call_timestamp,
                        "duration_seconds": call_duration,
                        "system_prompt": sys_msg,
                        "user_prompt": prompt,
                        "error": str(exc),
                        "temperature": payload.get("temperature"),
                        "top_p": payload.get("top_p"),
                        "max_tokens": payload.get("max_tokens"),
                        "success": False,
                    }
                )
            raise

    def __call__(self, prompt, system_prompt=None, **kwargs):
        return self.generate(prompt, system_prompt=system_prompt, **kwargs)


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(console)


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
    text = (
        text.replace("<IMPROVED>", "")
        .replace("</IMPROVED>", "")
        .strip()
    )
    return text


def clean_code(text: str) -> str:
    return strip_improved_tags(strip_code_fences(text))


def parse_task_ids(raw: str, all_ids: Iterable[str]) -> List[str]:
    if not raw or raw.lower() == "all":
        return list[str](all_ids)
    return [tid.strip() for tid in raw.split(",") if tid.strip()]


def save_json_log(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_best_code(
    code_var: Variable,
    score: float,
    metrics: Dict[str, Any],
    best_score: float,
    best_metrics: Dict[str, Any],
    best_code: str,
    task_id: str,
    step: int,
) -> Tuple[float, Dict[str, Any], str]:
    if score > best_score:
        new_best_score = score
        new_best_metrics = metrics
        new_best_code = str(code_var.value)
        logging.info("[%s] iter=%d updated best code (new score=%.4f)", task_id, step, new_best_score)
        return new_best_score, new_best_metrics, new_best_code
    return best_score, best_metrics, best_code


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
    except Exception as exc:
        return 0.0, {
            "task_id": task_id,
            "error": f"evaluation error: {exc}",
            "base_passed": 0,
            "base_total": 0,
            "plus_passed": 0,
            "plus_total": 0,
            "base_all_passed": False,
            "plus_all_passed": False,
            "combined_score": 0.0,
            "is_perfect": False,
        }

    base_status, base_details = result["base"]
    plus_status, plus_details = result["plus"]

    def summarize(status, details):
        try:
            if details is not None and len(details) > 0:
                arr = list[Any](details) if hasattr(details, "__iter__") else []
                passed = sum(1 for x in arr if x)
                total = len(arr)
                rate = passed / total if total else 0.0
                all_ok = status == PASS and passed == total
            else:
                passed, total = 0, 0
                rate = 1.0 if status == PASS else 0.0
                all_ok = status == PASS
        except Exception:
            passed, total = 0, 0
            rate = 1.0 if status == PASS else 0.0
            all_ok = status == PASS
        return passed, total, rate, all_ok

    base_passed, base_total, base_rate, base_ok = summarize(base_status, base_details)
    plus_passed, plus_total, plus_rate, plus_ok = summarize(plus_status, plus_details)
    combined = 0.5 * (base_rate + plus_rate)

    metrics = {
        "task_id": task_id,
        "base_passed": base_passed,
        "base_total": base_total,
        "plus_passed": plus_passed,
        "plus_total": plus_total,
        "base_pass_rate": base_rate,
        "plus_pass_rate": plus_rate,
        "base_all_passed": base_ok,
        "plus_all_passed": plus_ok,
        "combined_score": combined,
        "is_perfect": base_ok and plus_ok,
    }
    return combined, metrics


def optimize_task(
    task_id: str,
    problem: Dict,
    engine: TextGradEngine,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    workdir = Path(args.output_dir) / task_id.replace("/", "_")
    workdir.mkdir(parents=True, exist_ok=True)
    best_path = workdir / "best_program.py"
    log_path = workdir / "optimization_log.json"
    model_calls_log_path = workdir / "model_calls_log.json"

    optimization_logs: List[Dict[str, Any]] = []
    model_calls_log: List[Dict[str, Any]] = []

    if args.resume and best_path.exists():
        logging.info("[%s] found existing best_program.py, skipping optimization", task_id)
        code = best_path.read_text(encoding="utf-8")
        score, metrics = eval_code(task_id, code, problem, get_mbpp_plus_hash())
        return {
            "task_id": task_id,
            "solution": code,
            "metrics": metrics,
            "best_score": score,
            "from_cache": True,
        }

    init_prompt = build_initial_prompt(problem, args.generation_hint or "")
    raw_resp = engine.generate(init_prompt)
    init_code = strip_code_fences(raw_resp)
    code_var = Variable(
        init_code,
        requires_grad=True,
        role_description="MBPP candidate solution",
    )

    current_step_log: Dict[str, Any] = {}

    def log_callback(log_data: Dict[str, Any]) -> None:
        current_step_log.update(log_data)
        if log_data.get("success", False):
            model_calls_log.append(log_data)
            save_json_log(model_calls_log, model_calls_log_path)

    logging_engine = TextGradEngine(
        model=engine.model_string,
        api_key=getattr(engine.client, "api_key", ""),
        base_url=getattr(engine.client, "base_url", ""),
        temperature=engine.temperature,
        top_p=engine.top_p,
        max_tokens=engine.max_tokens,
        timeout=getattr(engine.client, "timeout", 300),
        max_retries=getattr(engine.client, "max_retries", 5),
        system_prompt=engine.system_prompt,
        log_callback=log_callback,
        nothink=getattr(engine, "nothink", False),
    )
    logging_engine.client = engine.client

    global_kb_dir = Path(args.output_dir).parent / "t_bfgs_kb_global"
    global_kb_dir.mkdir(parents=True, exist_ok=True)

    KB_MODE_CONFIG = {
        "use-and-store": (True, True),
        "use-only": (True, False),
        "store-only": (False, True),
        "none": (False, False),
    }
    kb_mode = getattr(args, "kb_mode", "use-only")
    if kb_mode not in KB_MODE_CONFIG:
        raise ValueError(f"Unknown KB mode: {kb_mode}, choices: {list(KB_MODE_CONFIG.keys())}")

    use_kb, store_kb = KB_MODE_CONFIG[kb_mode]

    kb: Optional[HessianProxyKB] = None
    if use_kb or store_kb:
        kb = HessianProxyKB(
            embedding_api_url=args.embedding_api_url,
            api_key=args.embedding_api_key,
            embedding_model=args.embedding_api_model,
            kb_path=str(global_kb_dir),
            collection_name="mbpp_t_bfgs_kb",
        )

    optimizer = TextualBFGS(
        parameters=[code_var],
        engine=logging_engine,
        kb_instance=kb if use_kb else None,
        domain="Code",
        verbose=1 if args.verbose else 0,
        enable_online_learning=store_kb,
        include_code_examples=getattr(args, "include_code_examples", False),
    )

    if kb_mode == "store-only" and kb is not None:
        optimizer._manual_kb = kb

    problem_hash = get_mbpp_plus_hash()
    init_code_cleaned = clean_code(init_code)
    best_score, best_metrics, best_code = -1.0, {}, init_code_cleaned
    code_var.value = init_code_cleaned

    for step in range(args.iterations):
        if step > 0 and best_code and best_score > -1.0:
            code_var.value = best_code
            logging.info("[%s] iter=%d restarting from best code (score=%.4f)", task_id, step, best_score)

        code_var.value = clean_code(str(code_var.value))
        score, metrics = eval_code(task_id, str(code_var.value), problem, problem_hash)

        best_score, best_metrics, best_code = update_best_code(
            code_var, score, metrics, best_score, best_metrics, best_code, task_id, step
        )

        logging.info(
            "[%s] iter=%d score=%.4f base=%.2f(%d/%d) plus=%.2f(%d/%d)%s",
            task_id,
            step,
            score,
            metrics.get("base_pass_rate", 0.0),
            metrics.get("base_passed", 0),
            metrics.get("base_total", 0),
            metrics.get("plus_pass_rate", 0.0),
            metrics.get("plus_passed", 0),
            metrics.get("plus_total", 0),
            " ✓" if metrics.get("is_perfect") else "",
        )
        if metrics.get("is_perfect"):
            break

        feedback = (
            f"Task {task_id} evaluation summary:\n"
            f"- Base cases: {metrics.get('base_passed', 0)}/{metrics.get('base_total', 0)}, "
            f"status={'pass' if metrics.get('base_all_passed') else 'fail'}\n"
            f"- Plus cases: {metrics.get('plus_passed', 0)}/{metrics.get('plus_total', 0)}, "
            f"status={'pass' if metrics.get('plus_all_passed') else 'fail'}\n"
            "Please fix the failing tests while keeping the function signature unchanged. "
            "Do not print extra text."
        )

        grad = Variable(feedback, requires_grad=False, role_description="test feedback")
        code_var.gradients = {grad}
        conversation_context = f"MBPP problem {task_id} description:\n{problem.get('prompt', '').rstrip()}"
        code_var.gradients_context = cast(
            Dict[Variable, Any],
            {
                grad: {
                    "context": conversation_context,
                    "variable_desc": code_var.role_description,
                    "response_desc": "candidate code",
                }
            },
        )

        current_step_log.clear()

        step_ok = False
        step_error: Optional[str] = None
        retry_count = 0
        for retry in range(max(1, args.max_retries)):
            retry_count = retry + 1
            try:
                optimizer.step()
                step_ok = True
                break
            except (AttributeError, IndexError, TypeError) as exc:
                step_error = str(exc)
                logging.warning(
                    "[%s] optimizer response parsing failed, retry %d/%d: %s",
                    task_id,
                    retry + 1,
                    args.max_retries,
                    exc,
                )

        user_prompt = current_step_log.get("user_prompt", "")
        system_prompt_used = current_step_log.get("system_prompt", getattr(optimizer, "system_prompt", ""))

        step_log: Dict[str, Any] = {
            "step": step,
            "task_id": task_id,
            "score_before": score,
            "metrics_before": metrics,
            "feedback": feedback,
            "user_prompt": user_prompt,
            "system_prompt": system_prompt_used,
            "response": current_step_log.get("response", ""),
            "temperature_used": current_step_log.get("temperature"),
            "top_p_used": current_step_log.get("top_p"),
            "max_tokens_used": current_step_log.get("max_tokens"),
            "call_timestamp": current_step_log.get("timestamp"),
            "call_duration_seconds": current_step_log.get("duration_seconds"),
            "step_success": step_ok,
            "step_error": step_error if not step_ok else None,
            "retry_count": retry_count,
        }

        if not step_ok:
            logging.error("[%s] optimizer failed repeatedly, skipping this step", task_id)
            optimization_logs.append(step_log)
            save_json_log(optimization_logs, log_path)
            continue

        code_var.value = clean_code(str(code_var.value))
        score_after, metrics_after = eval_code(task_id, str(code_var.value), problem, problem_hash)

        best_score, best_metrics, best_code = update_best_code(
            code_var, score_after, metrics_after, best_score, best_metrics, best_code, task_id, step
        )

        step_log.update(
            {
                "score_after": score_after,
                "metrics_after": metrics_after,
                "score_change": score_after - score,
                "code_after": str(code_var.value),
                "best_score_after_step": best_score,
            }
        )

        optimization_logs.append(step_log)
        save_json_log(optimization_logs, log_path)

        if metrics_after.get("is_perfect"):
            logging.info("[%s] iter=%d reached perfect score, stopping", task_id, step)
            break

    best_path.write_text(str(best_code), encoding="utf-8")

    summary_log = {
        "task_id": task_id,
        "total_steps": len(optimization_logs),
        "best_score": best_score,
        "best_metrics": best_metrics,
        "final_code": best_code,
    }
    summary_path = workdir / "optimization_summary.json"
    save_json_log(summary_log, summary_path)

    return {
        "task_id": task_id,
        "solution": best_code,
        "metrics": best_metrics,
        "best_score": best_score,
    }


def write_samples_jsonl(samples: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps({"task_id": item["task_id"], "solution": item["solution"]}) + "\n")


def run_evalplus(samples_path: Path, args: argparse.Namespace) -> None:
    print(f"\nRunning evalplus evaluation, samples: {samples_path}")
    evalplus_evaluate(
        dataset="mbpp",
        samples=str(samples_path),
        base_only=args.base_only,
        parallel=args.parallel,
        test_details=True,
        i_just_wanna_run=False,
    )


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run TextBFGS (Textual-BFGS) with TextGrad on MBPP/MBPP+."
    )
    parser.add_argument("--model", default="Qwen3-235B-A22B", help="model name")
    parser.add_argument("--api-key", default="Bearer YOUR_API_KEY_HERE", help="API key")
    parser.add_argument(
        "--base-url",
        default="https://api.example.com/v1",
        help="API base URL (without /chat/completions suffix)",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="top_p")
    parser.add_argument("--max-tokens", type=int, default=10240, help="max generated tokens")
    parser.add_argument("--max-retries", type=int, default=5, help="LLM generation retries")
    parser.add_argument("--request-timeout", type=int, default=300, help="single request timeout (seconds)")
    parser.add_argument("--iterations", type=int, default=20, help="TextGrad iterations per task")
    parser.add_argument("--task-ids", type=str, default="all", help="comma-separated task_id list or 'all'")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="output directory (default: derived from model name)",
    )
    parser.add_argument(
        "--samples-path",
        type=str,
        default=None,
        help="path to jsonl for evalplus (default: derived from model name)",
    )
    parser.add_argument("--parallel", type=int, default=None, help="evalplus processes")
    parser.add_argument("--base-only", type=bool, default=False, help="run only base cases in evalplus")
    parser.add_argument(
        "--skip-evalplus",
        type=bool,
        default=False,
        help="skip evalplus evaluation (only generate samples)",
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default="You are a senior Python engineer. Improve the target function to pass hidden tests.",
        help="system prompt for the model",
    )
    parser.add_argument(
        "--eval-results-path",
        type=str,
        default=None,
        help="path to eval_results.json used to identify failed tasks",
    )
    parser.add_argument(
        "--evolve-failed-only",
        type=bool,
        default=False,
        help="only iterate over tasks that previously failed evalplus",
    )
    parser.add_argument(
        "--resume",
        type=bool,
        default=True,
        help="skip optimization if best_program.py already exists for a task",
    )
    parser.add_argument(
        "--generation-hint",
        type=str,
        default="",
        help="extra hint appended to the initial generation prompt",
    )
    parser.add_argument(
        "--kb-mode",
        type=str,
        default="use-and-store",
        choices=["use-and-store", "use-only", "store-only", "none"],
        help="knowledge base mode: use-and-store, use-only, store-only, none",
    )
    parser.add_argument(
        "--embedding-api-url",
        type=str,
        default="https://api.example.com/embeddings",
        help="embedding API URL for Hessian-Proxy KB",
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default="YOUR_EMBEDDING_API_KEY_HERE",
        help="embedding API key for Hessian-Proxy KB",
    )
    parser.add_argument(
        "--embedding-api-model",
        type=str,
        default="Qwen3-Embedding-8B",
        help="embedding model name for Hessian-Proxy KB",
    )
    parser.add_argument(
        "--include-code-examples",
        action="store_true",
        default=False,
        help="if set, retrieve concrete code examples from the KB",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="print verbose optimizer logs",
    )
    parser.add_argument(
        "--nothink",
        action="store_true",
        default=True,
        help="disable internal reasoning mode in the model (if supported)",
    )
    args = parser.parse_args()

    model_name_safe = args.model.replace("/", "--").lower()
    model_name_for_file = args.model.replace("/", "--")
    kb_mode = getattr(args, "kb_mode", "use-only")
    include_code_examples = getattr(args, "include_code_examples", False)
    example_suffix = "_include_code_examples" if include_code_examples else ""

    if args.output_dir is None:
        args.output_dir = f"{model_name_safe}_runs/mbpp_t_bfgs_{kb_mode}{example_suffix}"

    if args.samples_path is None:
        args.samples_path = (
            f"evalplus_results/mbpp/{model_name_for_file}_t_bfgs_{kb_mode}{example_suffix}_samples.jsonl"
        )

    if args.eval_results_path is None:
        args.eval_results_path = (
            f"evalplus_results/mbpp/{model_name_for_file}.eval_results.json"
        )

    problems = get_mbpp_plus()
    failed_tasks: Dict[str, Dict[str, Any]] = {}
    all_original: Dict[str, Dict[str, Any]] = {}
    if args.evolve_failed_only and args.eval_results_path:
        if os.path.exists(args.eval_results_path):
            with open(args.eval_results_path, "r", encoding="utf-8") as f:
                eval_data: Any = json.load(f)
            eval_block = eval_data.get("eval", {})
            for tid, records in eval_block.items():
                if not records:
                    continue
                result = records[0]
                all_original[tid] = result
                if result.get("base_status") == "fail" or result.get("plus_status") == "fail":
                    failed_tasks[tid] = result

    if args.evolve_failed_only and failed_tasks:
        selected = list[str](failed_tasks.keys())
        print(f"Iterating only {len(selected)} failed tasks")
    else:
        selected = parse_task_ids(args.task_ids, problems.keys())
        print(f"Iterating {len(selected)} tasks")

    engine = TextGradEngine(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        system_prompt=args.system_message,
        nothink=args.nothink,
    )
    tg.set_backward_engine(engine, override=True)

    evolved: List[Dict[str, Any]] = []
    skipped = 0
    for tid in selected:
        if tid not in problems:
            print(f"Warning: task {tid} not in MBPP dataset, skipping")
            continue
        result = optimize_task(tid, problems[tid], engine, args)
        evolved.append(result)
        if result.get("from_cache"):
            skipped += 1

    if skipped:
        print(f"\nResumed run: {skipped} tasks loaded from cache, {len(evolved) - skipped} newly optimized")

    all_samples: List[Dict[str, str]] = []
    evolved_ids = {s["task_id"] for s in evolved}
    for sample in evolved:
        all_samples.append({"task_id": sample["task_id"], "solution": sample["solution"]})

    if args.evolve_failed_only and all_original:
        for task_id, origin in all_original.items():
            if task_id not in evolved_ids:
                all_samples.append({"task_id": task_id, "solution": origin.get("solution", "")})
        print(
            f"\nMerged results: {len(all_samples)} tasks "
            f"({len(evolved)} optimized + {len(all_samples) - len(evolved)} original correct)"
        )
    else:
        print(f"\nOptimization finished: {len(all_samples)} tasks (all TextGrad results)")

    samples_path = Path(args.samples_path)
    write_samples_jsonl(all_samples, samples_path)

    if not args.skip_evalplus:
        run_evalplus(samples_path, args)
    else:
        print("Skipped evalplus evaluation. You can run it manually, e.g.:")
        print(f"python -m evalplus.evaluate --dataset mbpp --samples {samples_path}")


if __name__ == "__main__":
    main()

