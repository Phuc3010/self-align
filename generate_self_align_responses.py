import math
import os
import fire
import time
import tqdm
import json
from vllm import LLM, SamplingParams
from pathlib import Path

def main(
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_seq_len: int = 3072,
    max_batch_size: int=32,
    generate_max_len: int = 256,
    num_devices: int = 1,
    input_file: str = None,
    output_file: str = None,
    meta_prompt_file: str = None,
):
    assert (
        input_file is not None and output_file is not None
    ), "Must specify input and output files"
    assert meta_prompt_file is not None, "Must specify meta prompt file"

    with open(input_file) as f:
        inputs = list(map(json.loads, f))
    
    print(len(inputs))

    generate_prompt_fn = generate_prompt
    
    generator = LLM(
        "mistralai/Mistral-7B-Instruct-v0.2",
        max_model_len=max_seq_len,
        tensor_parallel_size=num_devices,
    )
    tokenizer = generator.get_tokenizer()
    sampling_params = SamplingParams(n=2, top_p=top_p, max_tokens=generate_max_len, temperature=temperature)

    # record current progress

    if Path(output_file).exists():
        with open(output_file, "r") as f:
            outputs = f.readlines()
            outputs = [line for line in outputs if len(line.strip()) > 0]

    print("Skipping %d examples" % len(outputs))

    batching_inputs = tqdm.tqdm(
        BatchIterator(inputs, max_batch_size),
        desc="Batched inference",
    )

    output_handler = None
    output_handler = open(output_file, "a")

    # prepare inputs with batch size $max_batch_size
    all_prompts = []
    all_instructions = []
    all_inputs = []
    for iter, batched_inputs in enumerate(batching_inputs):
        prompts = [
            generate_prompt_fn(
                ex_input["instruction"].strip(), ex_input["input"].strip()
            )
            for ex_input in batched_inputs
        ]
        messages = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) for prompt in prompts]
        if iter == 0:
            print(messages[0])

        all_instructions.extend([ex_input['instruction'] for ex_input in batched_inputs])
        all_inputs.extend([ex_input['input'] for ex_input in batched_inputs])

        all_prompts.extend(messages)
        t1 = time.time()
    
    outputs = list(map(lambda x: [x.outputs[idx].text for idx in range(len(x.outputs))], 
                                generator.generate(all_prompts, sampling_params)))

    outputs = [[ele.replace("</s>", "") for ele in r] for r in outputs]
    print(outputs[0])

    if output_handler is not None:
        for instruction, input, output in zip(all_instructions, all_inputs, outputs):
            result = {
                "instruction": instruction,
                "input": input,
                "output_1": output[0],
                "output_2": output[1]
            }
            output_handler.write(json.dumps(result) + "\n")
        output_handler.flush()


class BatchIterator:
    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i : i + self.batch_size]

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)


def generate_prompt(instruction, input=None):
    if input:
        return f"""{instruction}\n\n{input}"""
    else:
        return f"""{instruction}"""

if __name__ == "__main__":
    fire.Fire(main)
