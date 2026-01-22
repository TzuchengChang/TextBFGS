"""T-BFGS (Textual-BFGS) optimizer implementation (anonymized snapshot).

This file is copied from our experimental repository and included in the
supplementary material to document the core algorithm used in TextBFGS.
It relies on the public TextGrad API and can be used as a drop-in optimizer.
"""

from textgrad.optimizer import Optimizer
from textgrad.optimizer.optimizer import get_gradient_and_context_text
from textgrad.variable import Variable
from textgrad import logger
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default

from .optimizer_prompts import T_BFGS_SYSTEM_PROMPT, T_BFGS_UPDATE_PROMPT
from .hessian_memory import HessianProxyKB

import re
from typing import List, Union


class TextualBFGS(Optimizer):
    """
    T-BFGS (Textual-BFGS) optimizer.

    High-level idea:
    - The optimizer treats LLM feedback as a \"textual gradient\".
    - A Hessian-Proxy Knowledge Base (HPKB) stores historical optimization
      trajectories (gradient, operator, before/after code) to approximate
      curvature information of the semantic space.
    - In each step, we perform a *one-pass* optimization:
        1. Retrieve similar historical patterns from HPKB.
        2. Ask the LLM to produce three sections in a single response:
           <GRADIENT>  : explanation / diagnosis of the error
           <OPERATOR>  : abstract natural-language rule describing the fix
           <IMPROVED>  : concrete corrected code
        3. Optionally store successful trajectories back into HPKB.
    """

    def __init__(
        self,
        parameters: List[Variable],
        engine: Union[EngineLM, str] = None,
        kb_instance: HessianProxyKB = None,
        domain: str = "General",
        verbose: int = 0,
        new_variable_tags: List[str] = None,
        enable_online_learning: bool = True,
        include_code_examples: bool = False,
    ):
        super().__init__(parameters)

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED>", "</IMPROVED>"]

        self.engine = validate_engine_or_get_default(engine)
        self.kb = kb_instance
        self.domain = domain
        self.verbose = verbose
        self.new_variable_tags = new_variable_tags
        self.enable_online_learning = enable_online_learning
        self.include_code_examples = include_code_examples

        # System prompt for the optimizer
        self.system_prompt = T_BFGS_SYSTEM_PROMPT

        if self.kb is None:
            logger.warning(
                "No HessianProxyKB instance provided. "
                "TextualBFGS will run without historical curvature information."
            )

        logger.info(
            "TextualBFGS initialized with domain=%s, kb=%s",
            domain,
            "enabled" if self.kb else "disabled",
        )

    def step(self) -> None:
        """
        Perform a single One-Pass optimization step.

        For each parameter:
        1. Aggregate the textual gradient.
        2. Retrieve curvature context from HPKB (if available).
        3. Build an update prompt and query the LLM once.
        4. Parse <GRADIENT>, <OPERATOR>, and <IMPROVED> sections.
        5. Update the parameter value and stage a pending trace for HPKB.
        """
        for parameter in self.parameters:
            if not parameter.gradients:
                logger.warning("No gradient found for %s", parameter.get_role_description())
                continue

            gradient_content = get_gradient_and_context_text(parameter)
            if isinstance(gradient_content, str):
                current_feedback = gradient_content
            elif isinstance(gradient_content, list):
                text_items = [item for item in gradient_content if isinstance(item, str)]
                current_feedback = "\n".join(text_items) if text_items else ""
            else:
                current_feedback = str(gradient_content)

            if not current_feedback or current_feedback.strip() == "":
                logger.warning("Empty gradient for %s", parameter.get_role_description())
                continue

            # 2. Retrieve curvature information from HPKB
            curvature_context = ""
            if self.kb:
                current_old_val = str(parameter.value) if self.include_code_examples else None
                curvature_context = self.kb.retrieve_inverse_hessian(
                    current_feedback,
                    self.domain,
                    top_k=3,
                    include_code_examples=self.include_code_examples,
                    current_old_val=current_old_val,
                )
            else:
                curvature_context = (
                    "### Optimization Manifold (Historical Gradients & Fixes):\n"
                    "No knowledge base available.\n"
                )

            # 3. Build the one-pass prompt
            prompt = T_BFGS_UPDATE_PROMPT.format(
                domain=self.domain,
                variable_desc=parameter.get_role_description(),
                variable_value=parameter.value,
                variable_grad=current_feedback,
                curvature_context=curvature_context,
            )

            logger.info(
                "T-BFGS prompt for %s", parameter.get_role_description(), extra={"prompt": prompt}
            )

            # 4. Single LLM call
            try:
                new_text = self.engine(prompt, system_prompt=self.system_prompt)
                logger.info(
                    "T-BFGS optimizer response", extra={"optimizer.response": new_text}
                )

                try:
                    gradient_match = re.search(
                        r"<GRADIENT>(.*?)</GRADIENT>", new_text, re.DOTALL
                    )
                    generated_gradient = (
                        gradient_match.group(1).strip()
                        if gradient_match
                        else current_feedback
                    )

                    operator_match = re.search(
                        r"<OPERATOR>(.*?)</OPERATOR>", new_text, re.DOTALL
                    )
                    generated_operator = (
                        operator_match.group(1).strip() if operator_match else None
                    )

                    improved_match = re.search(
                        r"<IMPROVED>(.*?)</IMPROVED>", new_text, re.DOTALL
                    )
                    if improved_match:
                        new_value = improved_match.group(1).strip()
                    else:
                        try:
                            new_value = (
                                new_text.split(self.new_variable_tags[0])[1]
                                .split(self.new_variable_tags[1])[0]
                                .strip()
                            )
                        except (IndexError, AttributeError):
                            logger.warning(
                                "Could not parse <IMPROVED> tag, using full response as fallback"
                            )
                            new_value = new_text.strip()

                    old_value = parameter.value

                    if generated_gradient and generated_gradient != current_feedback:
                        gradient_var = Variable(
                            generated_gradient,
                            requires_grad=False,
                            role_description="T-BFGS Generated Gradient (Internal Monologue)",
                        )
                        parameter.gradients.add(gradient_var)
                        logger.debug(
                            "Added generated gradient to parameter: %s...",
                            generated_gradient[:100],
                        )

                    if generated_operator:
                        logger.info(
                            "T-BFGS generated operator (abstract rule): %s...",
                            generated_operator[:200],
                        )
                    else:
                        logger.warning(
                            "T-BFGS did not generate <OPERATOR> section; "
                            "HPKB will fall back to auto-generated operators."
                        )

                    parameter.set_value(new_value)
                    logger.info(
                        "T-BFGS updated text", extra={"parameter.value": parameter.value}
                    )

                    if self.enable_online_learning and self.kb:
                        parameter._t_bfgs_pending_trace = {
                            "gradient_text": generated_gradient
                            if generated_gradient
                            else current_feedback,
                            "operator_text": generated_operator,
                            "old_val": old_value,
                            "new_val": new_value,
                            "domain": self.domain,
                        }

                    if self.verbose:
                        print("-----------------------TextualBFGS------------------------")
                        print(f"Domain: {self.domain}")
                        print(f"Variable: {parameter.get_role_description()}")
                        print(f"Updated value:\n{parameter.value}")
                        print("--------------------------------------------------------")

                except (IndexError, AttributeError) as exc:
                    logger.error(
                        "T-BFGS optimizer response could not be parsed",
                        extra={"optimizer.response": new_text, "error": str(exc)},
                    )
                    raise ValueError(
                        "T-BFGS optimizer response could not be parsed. "
                        "Expected format: <GRADIENT>...</GRADIENT>"
                        "<OPERATOR>...</OPERATOR><IMPROVED>...</IMPROVED>. "
                        "This can happen if the optimizer model cannot follow the instructions. "
                        f"Response (truncated): {new_text[:500]}..."
                    ) from exc

            except Exception as exc:  # noqa: BLE001
                logger.error("T-BFGS update failed: %s", exc, exc_info=True)
                raise

    def confirm_and_add_trace(
        self,
        parameter: Variable,
        score_before: float,
        score_after: float,
        min_improvement: float = 0.0,
    ) -> None:
        """
        Confirm and add a trajectory to HPKB only if the optimization improved the score.
        """
        if not self.enable_online_learning or not self.kb:
            return

        if not hasattr(parameter, "_t_bfgs_pending_trace"):
            logger.debug("No pending trace to confirm")
            return

        improvement = score_after - score_before
        if improvement <= min_improvement:
            logger.debug(
                "Skipping trace addition: improvement %.4f <= threshold %.4f "
                "(score: %.4f -> %.4f)",
                improvement,
                min_improvement,
                score_before,
                score_after,
            )
            delattr(parameter, "_t_bfgs_pending_trace")
            return

        trace_info = parameter._t_bfgs_pending_trace
        operator_text = trace_info.get("operator_text")

        self.kb.add_trace(
            gradient_text=trace_info["gradient_text"],
            old_val=trace_info["old_val"],
            new_val=trace_info["new_val"],
            domain=trace_info["domain"],
            operator_text=operator_text,
        )

        logger.info(
            "Confirmed and added trace to KB: improvement %.4f (score: %.4f -> %.4f)%s",
            improvement,
            score_before,
            score_after,
            (
                f", operator: {operator_text[:100]}..."
                if operator_text
                else ", using auto-generated operator"
            ),
        )

        delattr(parameter, "_t_bfgs_pending_trace")

