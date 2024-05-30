# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import json
import random
from collections import defaultdict
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HELPFUL_POSITIVE_AFFIXES = ['(giving a helpful response)']
HELPFUL_NEGATIVE_AFFIXES = ['(giving an unhelpful response)']

def _parse_indices_helper(indices):
    for index in indices.split(','):
        if '-' in index:
            start, end = index.split('-')
            for i in range(int(start), int(end) + 1):
                yield i
        else:
            yield int(index)

def parse_indices(indices):
    return list(_parse_indices_helper(indices))


def process_filter_result(result):
    result = result.replace('</s>', "") 
    return True, result

def generate_result(model, sampling_params, prompts, max_new_tokens=300):
    if type(prompts) == str:
        prompts = [prompts]
    
    all_results = [None for _ in range(len(prompts))]
    results = list(map(
        lambda x: x.outputs[0].text, model.generate(prompts, sampling_params=sampling_params)
    ))
    for i, result in enumerate(results):
        status, result = process_filter_result(result)
        all_results[i] = (status, result)
    return all_results


def create_prompts(prompt, tokenizer):
    positive_affix_choices = HELPFUL_POSITIVE_AFFIXES
    negative_affix_choices = HELPFUL_NEGATIVE_AFFIXES
    
    assert len(positive_affix_choices) == len(negative_affix_choices)
    index = random.choice(list(range(len(positive_affix_choices))))
    positive_affix = positive_affix_choices[index]
    negative_affix = negative_affix_choices[index]
    
    if prompt['input'] == '':
        instruction_entry = prompt['instruction']
    else:
        instruction_entry = prompt['instruction'] + '\n\n' + prompt['input']

    positive_prompt = tokenizer.apply_chat_template([{"role": "user", "content": instruction_entry}, {"role": "assistant", "content": f"{positive_affix}:"}], tokenize=False).replace("</s>", "")
    negative_prompt = tokenizer.apply_chat_template([{"role": "user", "content": instruction_entry}, {"role": "assistant", "content": f"{negative_affix}:"}], tokenize=False).replace("</s>", "")
    return ['chosen', 'rejected'], [positive_prompt, negative_prompt], index

def ready_for_analysis(args, analysis_waiting_entry):
    # check that we generated both outputs for the conversation prefix before moving on to post hoc scoring as needed
    if args.method == 'rlcd':
        return all([key in analysis_waiting_entry for key in ['chosen', 'rejected']])
    elif args.method in ['rlaif', 'rlcd_rescore']:
        return all([key in analysis_waiting_entry for key in ['resultA', 'resultB']])


def analyze_results(args, batch):
    if args.method == 'rlcd':
        return [{
            'prompt': entry.data['prompt'],
            'actual_prompts': entry.data['actual_prompts'],
            'chosen_result': entry.data['chosen'],
            'rejected_result': entry.data['rejected'],
            'status': True
        } for entry in batch]

class GenerationQueueEntry:
    def __init__(self, idx, key, retries, prompt, original_prompt, prompt_index):
        self.idx = idx
        self.key = key
        self.retries = retries
        self.prompt = prompt
        self.original_prompt = original_prompt
        self.prompt_index = prompt_index

class AnalysisQueueEntry:
    def __init__(self, idx, data, prompt_index):
        self.idx = idx
        self.data = data
        self.prompt_index = prompt_index


def main(args):
    with open(args.prompts_file) as f:
        prompts = list(map(json.loads, f))
    os.makedirs(args.out_dir, exist_ok=True)
    indices = parse_indices(args.indices) if args.indices is not None else list(range(len(prompts)))

    model = LLM(
        args.model_string,
        max_model_len=3072,
        tensor_parallel_size=1
    )
    sampling_params = SamplingParams(temperature=1.0, max_tokens=256)
    tokenizer = model.get_tokenizer()
    all_prompts = [create_prompts(prompt, tokenizer) for prompt in prompts]
    results = []
    for index, prompt in enumerate(all_prompts):
        keys, prompt_pair, idx = prompt
        entry_dict = {
            "prompt": prompts[index],
            "actual_prompts": prompt_pair,
            'status': True,
        }
        results.append(entry_dict)
    
    positive_prompts = [ele['actual_prompts'][0] for ele in results]
    negative_prompts = [ele['actual_prompts'][1] for ele in results]
    print(positive_prompts[0])
    print(positive_prompts[0][-1])
    print(negative_prompts[0])
    
    chosen_results = list(map(lambda x: x.outputs[0].text, model.generate(positive_prompts, sampling_params)))
    rejected_results = list(map(lambda x: x.outputs[0].text, model.generate(negative_prompts, sampling_params)))
    for idx, ele in enumerate(results):
        ele['chosen_result'] = chosen_results[idx].strip()
        ele['rejected_reult'] = rejected_results[idx].strip()

    print(results[0]['chosen_result'])
    
    for save_dict in results:
        with open(os.path.join(args.out_dir, 'mistral_7b_context_distill.json'), 'a') as f:
            f.write(json.dumps(save_dict) + "\n")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['harmless', 'helpful'])
    parser.add_argument('--method', type=str, choices=['rlcd', 'rlaif', 'rlcd_rescore'])
    parser.add_argument('--prompts-file', type=str)
    parser.add_argument('--model-string', type=str)
    parser.add_argument('--indices', type=str, default=None)
    parser.add_argument('--indices-remainder', type=int, default=0)
    parser.add_argument('--indices-mod', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-retries', type=int, default=5)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    main(args)

    