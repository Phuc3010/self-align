set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
export DATA_DIR="datasets"
export GPUS_PER_NODE=4
export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD:$PYTHONPATH"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    principle_synthetic_preference.py \
    --model_name $MODEL_NAME \
    --preferece_prompt "$DATA_DIR/prompts/synthetic_preference_prompt.json" \
    --rm_principles "$DATA_DIR/prompts/helpful_principle.json" \
    --response_pattern "$DATA_DIR/mistral_7b_self_align.json" \
    --output_file "$DATA_DIR/mistral_7b_principle_preference"