"""Reward functions for GRPO training."""

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig, parse, verify

def add_abcd(parsed_expres):
    added_expre = None
    for expre in parsed_expres:
        if isinstance(expre, str) and expre in 'abcdABCD' and len(expre) == 1:
            if expre in 'abcd':
                added_expre = expre.upper()
            else:
                added_expre = expre.lower()
            break
    if added_expre and added_expre not in parsed_expres:
        parsed_expres.append(added_expre)
    return parsed_expres


def compute_score(model_output: str, ground_truth: str):
    """Reward function that checks if the completion is the same as the ground truth."""
    if '\\boxed' not in ground_truth:
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    else:
        ground_truth_boxed = ground_truth

    gold_parsed = parse(
        ground_truth_boxed,
        extraction_mode="first_match",
        extraction_config=[
            # LatexExtractionConfig()
            LatexExtractionConfig(
                boxed_match_priority=0,
                normalization_config=NormalizationConfig
                    (
                        basic_latex=True,
                        units=True,
                        malformed_operators=True,
                        nits=True,
                        boxed="last",
                        equations=False,
                    )
            )
        ],
    )
    reward = 0.0
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            model_output,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            # breakpoint()
            gold_parsed = add_abcd(gold_parsed)
            answer_parsed = add_abcd(answer_parsed)
            reward = float(verify(answer_parsed, gold_parsed))
            # print({'predict': model_output[-30:], 'gt': ground_truth[-30:], 'reward': reward})
            # print({'predict': gold_parsed, 'gt': answer_parsed, 'reward': reward})
        except Exception as e:
            # print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
            reward = 0.0

    return reward