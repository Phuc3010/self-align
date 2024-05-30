python -u aggerate_synthetic_preference.py \
    --response_pattern "datasets/mistral_7b_self_align.json" \
    --preference_files "datasets/mistral_7b_principle_preference*.json" \
    --output_file "datasets/mistral7b_instruct_aggregated_preference.json" \
