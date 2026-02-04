
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -x


export PYTHONUNBUFFERED=1
export WANDB_API_KEY=your_wandb_api_key
export DEEPSEEK_API_KEY=your_deepseek_api_key
export monet_DEBUG=0

unset LD_PRELOAD
unset NCCL_TOPO_FILE
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=120
export VLLM_NO_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DASHBOARD=1
export RAY_DASHBOARD_ENABLED=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_NUM_CPUS=16
export RAY_NUM_GPUS=8
export USE_RAY_LOCAL=1
export RAY_ADDRESS=local
export RAY_METRICS_EXPORT_PORT=0
export RAY_LOG_TO_STDERR=0
export RAY_LOCAL_MODE=0
export RAY_task_exit_on_oom=1
export RAY_SPILL_DIR=/tmp/ray_spill
export RAY_TMPDIR=/tmp/ray_tmp
mkdir -p /tmp/ray_spill /tmp/ray_tmp
export RAY_OBJECT_STORE_MEMORY=134217728
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=300
conda easyr1
cd Monet/RL

MONET_RL_PATCH=1 # overwrite the transformers and vllm forward module
MODEL_PATH=path_to_your_model/Monet_SFT_stage3_model # path to your Monet SFT stage3 model
latent_size=10
export LATENT_SIZE=${latent_size}
ROLLOUT_N=8
TEMPERATURE=0.5
GPU_UTILIZATION=0.85
SELECT_ACC_THRESHOLD=0.6
KL_COEF=0.01
ORI_BSZ=64
ONLINE_ACCUM_SIZE=256
TRAIN_MAX_SAMPLES=-1
VAL_MAX_SAMPLES=-1
N_GPUS_PER_NODE=8
TENSOR_PARALLEL_SIZE=1
MONET_RL_SIGMA=10.0
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096

python -m verl.trainer.main \
    config=examples/config_monet.yaml \
    data.train_files=path_to_your_dataset/Thyme-RL/data \
    data.val_files=path_to_your_dataset/Thyme-RL/data/val \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=monet_latent${latent_size}_temp${TEMPERATURE}_tacc${SELECT_ACC_THRESHOLD}_rlsgm${MONET_RL_SIGMA} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.temperature=${TEMPERATURE} \
    worker.rollout.gpu_memory_utilization=${GPU_UTILIZATION} \
    worker.rollout.enable_chunked_prefill=true \
    worker.rollout.sampling_strategy=monet \
    worker.rollout.max_num_seqs=128 \
    worker.rollout.monet.select_acc_threshold=${SELECT_ACC_THRESHOLD} \
    worker.rollout.online_difficulty_sampling=true \
    worker.reward.reward_function=./examples/reward_function/monet_reward_function.py:compute_score_w_prev_correctness \
    worker.reward.repetition_penalty=true \
    worker.rule_based_judge.judge_function=./examples/reward_function/monet_reward_function.py:rule_then_api_batch_judge \
    worker.rule_based_judge.api_name="gemini-2.5-pro" \
    worker.actor.monet_rl_sigma=${MONET_RL_SIGMA} \
    worker.ref.monet_rl_sigma=${MONET_RL_SIGMA} \
    algorithm.kl_coef=${KL_COEF} \
    data.rollout_batch_size=${ORI_BSZ} \
    data.online_accum_size=${ONLINE_ACCUM_SIZE} \
    data.dataloader_num_workers=8 \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    

