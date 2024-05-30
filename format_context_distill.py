import json
import argparse
from tqdm import tqdm
from datasets import Dataset

if __name__=='__main__':
    with open("datasets/mistral_7b_context_distill.json") as f:
        dataset = list(map(json.loads, f))
    
    all_prompts, all_chosen, all_rejected = [], [], []
    for sample in tqdm(dataset):
        prompt = sample['prompt']['instruction']
        if sample['prompt']['input'] != '':
            prompt += '\n\n' + sample['prompt']['input']
        
        all_prompts.append(prompt)
        all_chosen.append(sample['chosen_result'])
        all_rejected.append(sample['rejected_reult'])
    
    ds_dict = {
        "prompt": all_prompts,
        "chosen": all_chosen,
        "rejected": all_rejected
    }
    ds = Dataset.from_dict(ds_dict)
    ds.save_to_disk("datasets/rlcd_context_distill")
