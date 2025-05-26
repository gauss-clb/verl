from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig, parse, verify
import sympy
import math

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


def cast_float(expre):
    if isinstance(expre, (sympy.core.numbers.Float, sympy.core.numbers.Integer)):
        return float(expre)
    if isinstance(expre, str):
        try:
            return float(expre)
        except:
            pass
    return None


def check_float_equal(answer_parsed, gold_parsed):
    answer, gold = None, None
    for expre in answer_parsed:
        answer = cast_float(expre)
        if answer:
            break
    for expre in gold_parsed:
        gold = cast_float(expre)
        if gold:
            break
    if answer and gold:
        return math.fabs(answer - gold) < 0.0001
    return None


def check_latex(model_output: str, ground_truth: str):
    """
        Reward function that checks if the completion is the same as the ground truth.
        Return: 
            True/False: with boxed
            None: without boxed
    """
    if '\\boxed' not in ground_truth:
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    else:
        ground_truth_boxed = ground_truth

    gold_parsed = parse(
        ground_truth_boxed,
        extraction_mode="first_match",
        extraction_config=[
            LatexExtractionConfig(
                boxed_match_priority=0,
                normalization_config=NormalizationConfig(
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
    if len(gold_parsed) == 0:
        return False
    
    # We require the answer to be provided in correct latex (no malformed operators)
    answer_parsed = parse(
        model_output,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    basic_latex=True,
                    units=True,
                    malformed_operators=True,
                    nits=True,
                    boxed="all",
                    equations=False,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
            )
        ],
        extraction_mode="first_match",
    )
    if len(answer_parsed) == 0:
        return None
    
    reward = False
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    try:
        gold_parsed = add_abcd(gold_parsed)
        answer_parsed = add_abcd(answer_parsed)
        is_float_equal = check_float_equal(answer_parsed, gold_parsed)
        if is_float_equal is None:
            reward = verify(answer_parsed, gold_parsed)
        else:
            reward = is_float_equal
        # print({'predict': model_output[-30:], 'gt': ground_truth[-30:], 'reward': reward})
        # print({'predict': gold_parsed, 'gt': answer_parsed, 'reward': reward})
    except Exception as e:
        # print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
        reward = False

    return reward


def check_latex_and_text(model_output: str, ground_truth: str):
    """
        Reward function that checks if the completion is the same as the ground truth.
        Return: 
            True/False
    """
    if '\\boxed' not in ground_truth:
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    else:
        ground_truth_boxed = ground_truth

    gold_parsed = parse(
        ground_truth_boxed,
        extraction_mode="first_match",
        extraction_config=[
            LatexExtractionConfig(
                boxed_match_priority=0,
                normalization_config=NormalizationConfig(
                    basic_latex=True,
                    units=True,
                    malformed_operators=True,
                    nits=True,
                    boxed="last",
                    equations=False,
                )
            ),
            ExprExtractionConfig()
        ],
    )
    if len(gold_parsed) == 0:
        return False
    
    reward = False
    # We require the answer to be provided in correct latex (no malformed operators)
    answer_parsed = parse(
        model_output,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    basic_latex=True,
                    units=True,
                    malformed_operators=True,
                    nits=True,
                    boxed="all",
                    equations=False,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
            StringExtractionConfig()
        ],
        extraction_mode="first_match",
    )
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    try:
        gold_parsed = add_abcd(gold_parsed)
        answer_parsed = add_abcd(answer_parsed)
        is_float_equal = check_float_equal(answer_parsed, gold_parsed)
        if is_float_equal is None:
            reward = verify(answer_parsed, gold_parsed)
        else:
            reward = is_float_equal
        # print({'predict': model_output[-30:], 'gt': ground_truth[-30:], 'reward': reward})
        # print({'predict': gold_parsed, 'gt': answer_parsed, 'reward': reward})
    except Exception as e:
        # print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
        reward = False

    return reward


if __name__ == '__main__':
    texts = ['\\boxed{35 cm}', 'final answer is 22', '22', 'choice A', 'the aswer is b', 'solution is 20.123456']
    answers = ['35', '22', '22', 'a', 'B', '20.123457']
    for text in texts:
        gold_parsed = parse(
            text,
            extraction_mode="first_match",
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0,
                    normalization_config=NormalizationConfig(
                        basic_latex=True,
                        units=True,
                        malformed_operators=True,
                        nits=True,
                        boxed="all",
                        equations=False,
                    ),
                ),
                ExprExtractionConfig(),
                StringExtractionConfig()
            ],
            parsing_timeout=1000000000
        )
        print(gold_parsed)

    for text, answer in zip(texts, answers):
        print(check_latex(text, answer), check_latex_and_text(text, answer))
    