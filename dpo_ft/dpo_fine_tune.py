import os
import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import Dataset, load_dataset
from utils.model_utils import load_tokenizer, load_model
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import json
import argparse

def load_dpo_train_data(args):
    if args.detection_method in ["fast-detectgpt"]:
        dpo_file_path = f"{args.dataset_file}/dpo_data/{args.base_model_name}.{args.detection_method}_scoring.{args.scoring_model_name}_reference.{args.reference_model_name}_dpo_train.json"
    if args.detection_method in ["roberta-large", "roberta-base"]:
        dpo_file_path = f"{args.dataset_file}/dpo_data/{args.base_model_name}.{args.detection_method}_dpo_train.json"
    with open(dpo_file_path, 'rb') as f:
        data_unprocessed = json.load(f)
    data = {}
    data['prompt']=[item for item in data_unprocessed['prompt']]
    data['chosen'] = [item for item in data_unprocessed['chosen']]
    data['rejected'] = [item for item in data_unprocessed['rejected']]
    return data


def process(row, tokenizer):
    row["prompt"] = tokenizer.apply_chat_template(row["prompt"], tokenize=False)
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row

def my_dataset(data):
    my_data = {"prompt": [], "chosen": [], "rejected": []}
    for i in range(len(data["prompt"])):
        p = data["prompt"][i]
        my_data["prompt"].append(p)

        c = data["chosen"][i]
        my_data["chosen"].append(c)

        r = data["rejected"][i]
        my_data["rejected"].append(r)
    my_data = Dataset.from_dict(my_data)
    return my_data



def my_dataset_by_data_builder(data, tokenizer):
    my_data = {"chosen": [], "rejected": [], "prompt": []}
    for i in range(len(data["prompt"])):
        x = data["prompt"][i]
        my_data["prompt"].append(x)

        y_1 = data["chosen"][i]
        message = [{'content': x, 'role': 'user'},
                   {'content': y_1, 'role': 'assistant'}]
        message = tokenizer.apply_chat_template(message, tokenize=False)
        my_data["chosen"].append(message)

        y_2 = data["rejected"][i]
        message = [{'content': x, 'role': 'user'},
                   {'content': y_2, 'role': 'assistant'}]
        message = tokenizer.apply_chat_template(message, tokenize=False)
        my_data["rejected"].append(message)
    my_data = Dataset.from_dict(my_data)
    return my_data



def experiment(args):
    # load model
    tokenizer = load_tokenizer(args.base_model_name, args.dataset, args.cache_dir,use_ft=False)
    base_model = load_model(args, args.base_model_name, args.device, args.cache_dir,use_ft=False,data_generation_flag=False)

    dpo_train_data=load_dpo_train_data(args)
    #dataset
    ds = my_dataset(dpo_train_data)
    original_columns = ds.column_names

    def chatml_format(example):
        # Format system
        system = ""

        # Format instruction
        message = {"role": "user", "content": example['prompt']}
        prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

        # Format chosen answer
        chosen = example['chosen'] + "<|im_end|>\n"

        # Format rejected answer
        rejected = example['rejected'] + "<|im_end|>\n"

        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    # Format dataset
    dataset = ds.map(
        chatml_format,
        remove_columns=original_columns
    )

    # LoRA configuration
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Model to fine-tune
    base_model = get_peft_model(base_model, peft_config)

    training_args = DPOConfig(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=args.epoch,
        save_steps=100,
        learning_rate=2e-4,
        bf16=True,
        save_total_limit=3,
        logging_steps=1,
        output_dir=f"{args.cache_dir}/dpo_logging",
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False
    )

    dpo_trainer = DPOTrainer(
        model=base_model,
        ref_model=None,
        args=training_args,
        beta=args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=10,
        max_length=args.max_length,
        max_target_length=args.max_length
    )
    dpo_trainer.train()

    dpo_dir = os.path.join(args.dpo_dir, args.base_model_name)
    dpo_dir = os.path.join(dpo_dir, args.dataset)
    if args.detection_method in ["fast-detectgpt"]:
        dpo_savepath=f"{args.detection_method}_scoring_{args.scoring_model_name}_reference_{args.reference_model_name}"
    if args.detection_method in ["roberta-large", "roberta-base"]:
        dpo_savepath =f"{args.detection_method}"
    dpo_dir=os.path.join(dpo_dir,dpo_savepath)
    dpo_param_savepath=f"beta_{args.beta}_ep_{args.epoch}"
    dpo_dir = os.path.join(dpo_dir, dpo_param_savepath)
    dpo_trainer.model.save_pretrained(dpo_dir)
    tokenizer.save_pretrained(dpo_dir)

    print("complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="openwebtext")
    parser.add_argument('--dataset_file', type=str, default="../../../detectionfiles/openwebtext")
    parser.add_argument('--detection_method', type=str, default="fast-detectgpt") #roberta-large, fast-detectgpt
    parser.add_argument('--reference_model_name', type=str, default="llama2-7b")  # gpt-j-6B, llama2-7b
    parser.add_argument('--scoring_model_name', type=str, default="llama2-7b")  # gpt-neo-2.7B, llama2-7b
    parser.add_argument('--base_model_name', type=str, default="llama2-7b")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=200)

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../../../detectionfiles/cache")
    parser.add_argument('--dpo_dir', type=str,
                        default="../../../detectionfiles/dpo_checkpoint")
    args = parser.parse_args()
    experiment(args)