"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from .reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from .math_check import check_latex
from ..file_utils import write_jsonl, read_text, read_jsonl, get_signal

import os
import re

THOUGHT_DELIMITER_END = "</think>"

class MathVerifyFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __init__(self, config: RewardConfig, eos_token: str):
        super().__init__(config)
        self.eos_token = eos_token

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        model_response = input.model_response
        
        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response and self.eos_token in model_response:
            match = re.match(r".+?</think>(.+)", model_response, re.DOTALL)
            if match:
                model_solution = match.group(1).strip()
            else:
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False, extra_info={"format_error": "regex unmatch"})
        else:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False, extra_info={"format_error": "no <think> or eos_token"})

        # Process the ground truth
        ground_truth = input.metadata.get("answer", None)
        if ground_truth is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False, extra_info={"ground_truth": "none"})
        if isinstance(ground_truth, (list, tuple)):
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False, extra_info={"ground_truth": "list or tuple"})

        ground_truth = str(ground_truth)
        score = 1.0 if check_latex(model_solution, ground_truth) else 0.0
        if score > 0:
            reward = self.config.correct_reward
            if input.metadata.get("has_toolcall", False):
                reward += self.config.toolcall_bonus
            return RewardOutput(reward=reward, is_correct=True)
       
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False, extra_info={"match": "false"})


def rllm_math_reward_fn(data_source: str, llm_solution: str, ground_truth: Union[str, List[str]], extra_info={}):
    """Evaluates mathematical solutions against ground truth answers.

    This function creates a reward function to evaluate mathematical solutions by comparing
    them against provided ground truth answers. It can optionally use a language model
    for more sophisticated answer validation.

    Args:
        data_source: The source/dataset the problem comes from
        llm_solution: The solution string provided by the language model to evaluate
        ground_truth: Either a single string or list of strings containing valid answers
        enable_llm: Whether to enable language model validation for complex cases (default: False)

    Returns:
        bool: True if the solution is deemed correct, False otherwise

    Example:
        >>> rllm_math_reward_fn("gsm8k", "x = 5", "5", False)
        True
    """
    # print(f'***** extra_info: {extra_info}')
    reward_config = RewardConfig()
    reward_fn = MathVerifyFn(reward_config, eos_token=extra_info.get('eos_token', '<｜end▁of▁sentence｜>'))
    reward_response = reward_fn(RewardInput(problem=None,
                                            problem_type=RewardType.MATH,
                                            model_response=llm_solution,
                                            metadata={"answer": ground_truth, **extra_info},
                                            data_source=data_source))
    
    
    if extra_info.get('is_save', False):
        assert 'default_local_dir' in extra_info, 'default_local_dir must be provided'
        default_local_dir = extra_info.get('default_local_dir', '')
        global_steps = extra_info.get('global_steps', -1)
        status = extra_info.get('status', 'train')
        signals = get_signal(default_local_dir)
        if 'is_verify_save' in signals and status == 'train':
            path = os.path.join(default_local_dir, 'rllm_verify.jsonl')
            items = [{
                "global_step": global_steps,
                "ground_truth": ground_truth,
                "input": "",
                "output": llm_solution,
                "reward": reward_response.reward,
                "extra_info": reward_response.extra_info
            }]
            write_jsonl(items, path, mode='a')
        elif 'is_validate_save' in signals and status == 'validate':
            path = os.path.join(default_local_dir, 'rllm_aime.jsonl')
            items = [{
                "global_step": global_steps,
                "ground_truth": ground_truth,
                "input": "",
                "output": llm_solution,
                "reward": reward_response.reward,
                "extra_info": reward_response.extra_info
            }]
            write_jsonl(items, path, mode='a')
    # print(reward_response)
    return reward_response.reward


if __name__ == "__main__":
    # reward = MathVerifyFn(RewardConfig(), eos_token='<｜end▁of▁sentence｜>')
    # test_input = RewardInput(
    #     data_source="",
    #     problem=(
    #         "Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots "
    #         "$r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots "
    #         "$r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient "
    #         "of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, "
    #         "leaving your answer in terms of $x$. (You may assume that $x$ is not equal to "
    #         "any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$."
    #     ),
    #     problem_type=RewardType.MATH,
    #     model_response=(
    #         "<think>...</think>\nThe answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.<｜end▁of▁sentence｜>"
    #     ),
    #     metadata={"answer": "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$", "has_toolcall": False}
    # )
    # output = reward(test_input)
    # print(output)


    items = read_jsonl('test_case.jsonl')
    # data_source = ''
    llm_solution = items[0]['output']
    ground_truth = items[0]['ground_truth']

    # print(rllm_math_reward_fn(data_source, llm_solution, ground_truth))
    kwargs = {'reward_type': 'math_verify', 'eos_token': '<｜end▁of▁sentence｜>'}
    reward = rllm_math_reward_fn('', llm_solution, ground_truth, **kwargs)
    # print(reward)

