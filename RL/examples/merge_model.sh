cd Monet/RL
conda activate easyr1
CKPT_PATH=Monet/RL/run_name/global_step_xxx/actor
python3 -m scripts.model_merger --local_dir=${CKPT_PATH}
