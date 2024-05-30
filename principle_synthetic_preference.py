import fcntl
import glob
import json
import os
import random

import numpy as np
import torch
from accelerate.utils import gather_object, gather
import tqdm
import fire

from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import load_dataset
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
import warnings
from accelerate import Accelerator
warnings.filterwarnings("ignore")

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))


def main(
    model_name: str,
    preferece_prompt: str,
    rm_principles: str,
    response_pattern: str,
    output_file: str,
    tokenizer_name: str = "TheBloke/dromedary-65b-lora-HF",  # a random llama-based model
):
    with open(preferece_prompt) as f:
        PREFERENCE_META_PROMPT = list(map(json.loads, f))

    with open(rm_principles, "r") as f:
        principle_definitions = json.load(f)

    data_file = response_pattern
    cache = ""
    with open(data_file) as f:
        data = list(map(json.loads, f))
    
    data = sorted(data, key=lambda x: x["instruciton"])

    preference_dataset = []

    for sample in data:
        if (
            len(sample["output_1"]) < 16
            or len(sample["output_2"]) < 16
        ):
            continue

        preference_dataset.append(
            {
                "instruction": sample["instruciton"],
                "input": sample["input"],
                "output_1": sample["output_1"].strip(),
                "output_2": sample["output_2"].strip(),
                "preference": 0,
            }
        )

    random.Random(42).shuffle(preference_dataset)

    print(f"Starting to load the model {model_name} into memory")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(world_size)
    torch.cuda.set_device(rank)

    output_file = output_file + f"{rank}.json" 
    print(preference_dataset[0 + rank])

    batch_size = 4 # single 80GB GPU

    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
        attn_implementation='flash_attention_2'
    )
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left", truncation_side="left", model_max_length=1536)
    tok.pad_token = tok.eos_token

    print(f"Successfully loaded the model {model_name} into memory")
    print_flag = True

    idxs = []
    dimensions = []
    prompts = []
    preferences = []

    for idx in tqdm.tqdm(range(len(preference_dataset))):
        if idx % world_size != rank:
            continue

        for principle_definition in principle_definitions:
            dimension = principle_definition["dimension"]
            definition = principle_definition["definition"]

            data = preference_dataset[idx]
            instruction = data["instruction"]
            instruction_input = data["input"]

            if instruction_input:
                instruction += "\n\n" + instruction_input

            output_1 = data["output_1"]
            output_2 = data["output_2"]
            preference = data["preference"]
            preferences.append(preference)
            idxs.append(idx)
            dimensions.append(dimension)

            for a, b in [(output_1, output_2), (output_2, output_1)]:
                prompt_for_score = PREFERENCE_META_PROMPT[0]['content'].format(
                    UserInstruction=instruction,
                    OutputA=a,
                    OutputB=b,
                    Dimension=dimension,
                    Definition=definition,
                )

                chat_prompt = tok.apply_chat_template(
                    [
                        {"role": "user", "content": prompt_for_score},
                        {"role": "assistant", "content": PREFERENCE_META_PROMPT[1]['content'].format(Dimension=dimension)}
                    ], tokenize=False
                ).replace("</s>", "").strip()

                if print_flag:
                    print_flag = False
                
                prompts.append(chat_prompt)

            if len(prompts) == batch_size * 2:
                tokenized_input = tok(
                    prompts,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding="max_length",
                    return_attention_mask=True,
                    truncation=True,
                )

                input_ids = tokenized_input["input_ids"].to(m.device)
                attention_mask = tokenized_input["attention_mask"].to(m.device)

                with torch.inference_mode():
                    output = m(
                        input_ids,
                        attention_mask=attention_mask,
                    )

                logits = output.logits

                token_id_a = tok.encode("\n (a", add_special_tokens=False)[-1]
                token_id_b = tok.encode("\n (b", add_special_tokens=False)[-1]

                relative_scores = []
                prompt_preferences = []
                for ex_idx in range(batch_size):
                    score_a_for_1_2 = logits[ex_idx * 2 + 0, -1, token_id_a]
                    score_b_for_1_2 = logits[ex_idx * 2 + 0, -1, token_id_b]
                    score_a_for_2_1 = logits[ex_idx * 2 + 1, -1, token_id_a]
                    score_b_for_2_1 = logits[ex_idx * 2 + 1, -1, token_id_b]

                    relative_score_1_2 = (score_a_for_1_2 - score_b_for_1_2).item()
                    relative_score_2_1 = (score_b_for_2_1 - score_a_for_2_1).item()

                    if relative_score_1_2 > 0.0 and relative_score_2_1 > 0.0:
                        prompt_preference = 1
                    elif relative_score_1_2 < 0.0 and relative_score_2_1 < 0.0:
                        prompt_preference = 2
                    else:
                        prompt_preference = 0

                    relative_scores.append((relative_score_1_2, relative_score_2_1))
                    prompt_preferences.append(prompt_preference)

                outputs = []
                
                for ex_idx in range(batch_size):
                    outputs.append(
                        {
                            "example_idx": idxs[ex_idx],
                            "dimension": dimensions[ex_idx],
                            "preference": preferences[ex_idx],
                            "prompt_preference": prompt_preferences[ex_idx],
                            "relative_score": relative_scores[ex_idx],
                        }
                    )
                
                with open(output_file, "a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    for output in outputs:
                        f.write(json.dumps(output) + "\n")
                    fcntl.flock(f, fcntl.LOCK_UN)

                idxs = []
                dimensions = []
                prompts = []
                preferences = []


if __name__ == "__main__":
    fire.Fire(main)