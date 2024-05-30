from trl import DPOTrainer, create_reference_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_from_disk, load_dataset
import torch

def apply_chat_template(
    example,
    tokenizer
):
    example['prompt'] = tokenizer.apply_chat_template([{"role": "user", "content": example['prompt']}], tokenize=False)
    example['chosen'] = example['chosen'] + tokenizer.eos_token
    example['rejected'] = example['rejected'] + tokenizer.eos_token
    return example

if __name__=='__main__':
    raw_datasets = load_dataset("DatPySci/mistral7b-context-distill")
    train_dataset= raw_datasets['train']
    train_datset = train_dataset.shuffle(seed=42)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer
        },
    )

    args = TrainingArguments(
        output_dir='results/mistral7b_rlcd',
        gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        gradient_checkpointing_kwargs={"use_reetrant": False},
        do_eval=False,
        eval_strategy='no',
        learning_rate=5e-7,
        optim='rmsprop',
        run_name='mistral7b_rlcd',
        warmup_ratio=0.1,
        lr_scheduler_type='linear'
    )

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }

    model = "mistralai/Mistral-7B-Instruct-v0.2"
    ref_model = model

    ref_model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }

    trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        tokenizer=tokenizer,
        args=args,
        beta=0.1,
        max_length=512,
        max_prompt_length=256,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
