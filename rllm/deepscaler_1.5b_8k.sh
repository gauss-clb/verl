#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.

NNODES=${NNODES:=$(ray list nodes --filter "IS_HEAD_NODE=False" --filter "STATE=ALIVE" --format yaml | grep -c node_ip)}

model_path=${model_path:=/home/chenliangbo/models/DeepSeek-R1-Distill-Qwen-1.5B}
train_files=${train_files:=/home/chenliangbo/data/math_rl/deepscaler.parquet}
val_files=${val_files:=/home/chenliangbo/data/aime_rl/aime_2024.parquet}
experiment_name=${experiment_name:=deepscaler_1.5b_8k}
save_freq=${save_freq:=10000}
test_freq=${test_freq:=10000}
n_val=${n_val:=8}
rollout_n=${rollout_n:=8}
n_gpus_per_node=${n_gpus_per_node:=8}
log_val_generations=${log_val_generations:=0}
logger=${logger:="['console','wandb']"}
default_local_dir=${default_local_dir:=/home/chenliangbo/models/verl_checkpoints/test}
is_save=${is_save:=False}
val_before_train=${val_before_train:=True}
max_response_length=${max_response_length:=8192}
# reward_type=${reward_type:=default}
project_name=${project_name:=deepscaler}
tp=${tp:=1}
sp=${sp:=1}
train_batch_size=${train_batch_size:=128}
temperature=${temperature:=0.6}
learning_rate=${learning_rate:=1e-6}
ppo_micro_batch_size=${ppo_micro_batch_size:=64}
ppo_micro_batch_size=$((ppo_micro_batch_size * rollout_n))
ppo_mini_batch_size=${ppo_mini_batch_size:=64}
use_kl_loss=${use_kl_loss:=True}
entropy_coeff=${entropy_coeff:=0}
clip_ratio_low=${clip_ratio_low:=0.2}
clip_ratio_high=${clip_ratio_high:=0.2}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${ppo_micro_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${n_val} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=${logger} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=${val_before_train} \
    trainer.log_val_generations=${log_val_generations} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${default_local_dir} \
    trainer.total_epochs=30 \
    +trainer.is_save=${is_save}