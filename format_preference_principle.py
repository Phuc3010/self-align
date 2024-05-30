from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__=="__main__":
    data = []
    with open("datasets/mistral7b_instruct_aggregated_preference.json") as f:
        sample = f.read()
        data = json.loads(sample)
        # for line in f.readlines():
        #     data = json.loads(line)
    print(data[0].keys())
    from functools import reduce

    sanity = [ele['preference'] for ele in data]
    all_prompts, all_chosen, all_rejected = [], [], []
    
    for sample in data:
        prompt = sample['instruction']
        if sample['input'] != '':
            prompt += '\n\n' + sample['input']

        all_prompts.append(prompt)
        preference = sample['preference']
        if preference == 1:
            all_chosen.append(sample['output_1'])
            all_rejected.append(sample['output_2'])
        else:
            all_chosen.append(sample['output_2'])
            all_rejected.append(sample['output_1'])
    
    principle_ds = {
        "prompt": all_prompts,
        "chosen": all_chosen,
        "rejected": all_rejected
    }
    ds = Dataset.from_dict(principle_ds)
    ds.save_to_disk("datasets/principle_preference")