#!/usr/bin/env bash
# One cache-fill run plus four measured fused/response-only combinations.
set -Eeuo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3-8B model}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the training parquet file}"
VAL_FILE=${VAL_FILE:-$TRAIN_FILE}
PROFILE=${PROFILE:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
CACHE_ROOT=${CACHE_ROOT:-$PWD/rlmh-cache/$RUN_ID}
LOG_DIR=${LOG_DIR:-$PWD/rlmh-combination-logs/$RUN_ID}
export DEVICE=gpu VERL_USE_UV=0 RAY_DEDUP_LOGS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
mkdir -p "$CACHE_ROOT" "$LOG_DIR"

COMMON=(
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "data.train_files=$TRAIN_FILE"
  "data.val_files=$VAL_FILE"
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  data.train_batch_size=1
  data.gen_batch_size=1
  data.shuffle=false
  data.seed=42
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.actor.megatron.vanilla_mbridge=false
  actor_rollout_ref.actor.megatron.use_remove_padding=true
  actor_rollout_ref.actor.megatron.sequence_parallel=true
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2
  actor_rollout_ref.actor.megatron.context_parallel_size=2
  actor_rollout_ref.ref.megatron.sequence_parallel=true
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2
  actor_rollout_ref.ref.megatron.context_parallel_size=2
  "actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD:-true}"
  "actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD:-true}"
  "actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD:-true}"
  actor_rollout_ref.actor.ppo_mini_batch_size=1
  actor_rollout_ref.actor.data_loader_seed=42
  actor_rollout_ref.actor.optim.lr=0.0
  # Exercise backward even when every NIAH completion receives reward 1.
  actor_rollout_ref.actor.calculate_entropy=true
  actor_rollout_ref.actor.entropy_coeff=0.001
  actor_rollout_ref.rollout.n=2
  actor_rollout_ref.rollout.seed=42
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  actor_rollout_ref.rollout.ignore_eos=true
  "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
  trainer.use_v1=true
  trainer.v1.trainer_mode=sync
  trainer.total_training_steps=4
  trainer.resume_mode=disable
  trainer.val_before_train=false
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  'trainer.logger=["console"]'
  trainer.project_name=rlmh_combinations_a100
  trainer.experiment_name=rlmh_combinations_a100
  trainer.save_freq=-1
  trainer.test_freq=-1
  skip.rollout.enable=false
  skip.rollout_tq.enable=true
  "skip.rollout_tq.dump_dir=$CACHE_ROOT"
  'skip.rollout_tq.steps=[1,2,3,4]'
  skip.rollout_tq.action=cache
)

run_case() {
  local name=$1 fused=$2 enabled=$3 profile=$4
  shift 4
  bash examples/grpo_trainer/run_qwen3_8b_megatron.sh \
    "${COMMON[@]}" "$@" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_RESPONSE_ONLY_LM_HEAD_PROFILE=\"$profile\"" \
    "actor_rollout_ref.model.use_fused_kernels=$fused" \
    "actor_rollout_ref.actor.megatron.response_only_lm_head=$enabled" \
    "actor_rollout_ref.ref.megatron.response_only_lm_head=$enabled" \
    2>&1 | tee "$LOG_DIR/$name.log"
}

run_case 01-cache false false 0 "$@"
run_case 02-unfused-baseline false false "$PROFILE" "$@"
run_case 03-unfused-response-only false true "$PROFILE" "$@"
run_case 04-fused-baseline true false "$PROFILE" "$@"
run_case 05-fused-response-only true true "$PROFILE" "$@"

if [[ $PROFILE == 1 ]]; then
  python examples/profile/summarize_response_only_lm_head.py \
    --baseline "$LOG_DIR/02-unfused-baseline.log" \
    --optimized "$LOG_DIR/03-unfused-response-only.log" \
    --warmup 2 | tee "$LOG_DIR/summary-unfused.log"
  python examples/profile/summarize_response_only_lm_head.py \
    --baseline "$LOG_DIR/04-fused-baseline.log" \
    --optimized "$LOG_DIR/05-fused-response-only.log" \
    --warmup 2 | tee "$LOG_DIR/summary-fused.log"
fi
echo "Logs: $LOG_DIR"
echo "Cache: $CACHE_ROOT"
