#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${GRAPHGPO_RUNTIME_DIR:-/scratch/rai/vast1/u6066038/verl-agent-src.ArushA}"
ENV_DIR="${GRAPHGPO_ENV_DIR:-/scratch/rai/vast1/u6066038/envs/graphgpo}"
DATA_DIR="${GRAPHGPO_DATA_DIR:-$PROJECT_DIR/data/verl-agent}"
RUN_DIR="${GRAPHGPO_RUN_DIR:-$PROJECT_DIR/runs/webshop-smoke}"

export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR:$RUNTIME_DIR${PYTHONPATH:+:$PYTHONPATH}"
export JAVA_HOME="$ENV_DIR/lib/jvm"
export JVM_PATH="$ENV_DIR/lib/jvm/lib/server/libjvm.so"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR/.cache/wandb}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$DATA_DIR" "$RUN_DIR" "$HF_HOME" "$WANDB_DIR"

python -m examples.data_preprocess.prepare \
  --mode text \
  --local_dir "$DATA_DIR" \
  --train_data_size 1 \
  --val_data_size 1

cd "$RUNTIME_DIR"
python -m recipe.GraphGPO.main_graphgpo \
  algorithm.adv_estimator=graphgpo \
  algorithm.gamma=0.20 \
  algorithm.graphgpo.step_advantage_w=1.0 \
  algorithm.graphgpo.episode_advantage_w=1.0 \
  algorithm.graphgpo.mode=mean_std_norm \
  algorithm.graphgpo.normalize_distance=False \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA_DIR/text/train.parquet" \
  data.val_files="$DATA_DIR/text/test.parquet" \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=5120 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.10 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  env.env_name=webshop/WebAgentTextEnv \
  env.seed=0 \
  env.max_steps=5 \
  env.webshop.observation_mode=text_rich \
  env.rollout.n=2 \
  env.resources_per_worker.num_cpus=0.1 \
  ray_init.num_cpus=8 \
  trainer.critic_warmup=0 \
  trainer.logger='[console]' \
  trainer.project_name=verl_agent_webshop \
  trainer.experiment_name=graphgpo_qwen2.5_1.5b_webshop_smoke \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.default_local_dir="$RUN_DIR" \
  trainer.val_before_train=False \
  "$@"
