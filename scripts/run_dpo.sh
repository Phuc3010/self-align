export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero3.yaml --num_processes=7 run_dpo.py
