from verl.utils.reward_score.math_check import check_latex, check_latex_and_text
import math


def format_reward(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    '''
        是否以eos_token结束
        是否带有think标签
        是否带boxed标签，取决于指令
    '''
    # print(f'format_reward: {extra_info}')
    reward = 0
    eos_token = extra_info.get('eos_token', '<｜end▁of▁sentence｜>')
    if eos_token in solution_str:
        reward += 0.1
    if '</think>' in solution_str:
        reward += 0.05
    if '\\boxed' in solution_str:
        reward += 0.1
    return reward


def cosine_scaled_reward(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    # 带eos_token且答案正确，鼓励越短越好

    """Reward function that scales based on completion length using a cosine schedule.

    Shorter correct solutions are rewarded more than longer ones.
    Longer incorrect solutions are penalized less than shorter ones.

    Args:
        solution_str: model completion
        ground_truth: ground truth solution

    This function is parameterized by the following arguments:
        min_value_correct: Minimum reward for correct answers
        max_value_correct: Maximum reward for correct answers
        max_len: Maximum length for scaling
    """
    # print(f'cosine_scaled_reward: {extra_info}')
    max_value_correct = extra_info['max_value_correct']
    min_value_correct = extra_info['min_value_correct']
    reward_correct = extra_info['reward_correct']
    lower_bound_len = extra_info['lower_bound_len']
    max_len = extra_info['max_len']
    gen_len = extra_info['gen_len']

    is_correct = check_latex(solution_str, ground_truth)
    if is_correct: # 带boxed
        if gen_len <= lower_bound_len:
            reward = reward_correct + max_value_correct - min_value_correct
        else:
            # Apply cosine scaling based on length
            cosine = math.cos(gen_len / max_len * math.pi)
            reward = reward_correct + 0.5 * (max_value_correct - min_value_correct) * (1.0 + cosine)
    elif is_correct == False:
        reward = 0
    else: # 不带boxed
        is_correct = check_latex_and_text(solution_str, ground_truth)
        if is_correct:
            reward = reward_correct
        else:
            reward = 0

    return reward

if __name__ == '__main__':
    solution_list = ['************************* \\boxed{35 cm}', '\\boxed{35 cm} <｜end▁of▁sentence｜>', '\\boxed{35 cm}', '</think> final answer is 22', '\\boxed{22}', 'choice A', 'the answer is b', 'solution is 20.123456 <｜end▁of▁sentence｜>']
    ground_truth_list = ['35', '35', '36', '25', '23', 'a', 'B', '20.123457']
    extra_info = {'max_len': 100}
    for solution_str, ground_truth in zip(solution_list, ground_truth_list):
        extra_info['gen_len'] = len(solution_str)
        reward = cosine_scaled_reward('', solution_str, ground_truth, extra_info)
        reward2 = format_reward('', solution_str, ground_truth, extra_info)
        print(f'sol: {solution_str}, gt: {ground_truth}, reward: {reward}, format_reward: {reward2}')