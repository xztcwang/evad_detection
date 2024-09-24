from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, AutoPeftModelForCausalLM
import torch
import time
import os

# predefined models
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     't5-base': 'google-t5/t5-base',
                     't5-3b': 'google-t5/t5-3b',
                     'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
                     'llama-13b': 'huggyllama/llama-13b',
                     'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-13b': 'facebook/opt-13b',
                     'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
                     'llama3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
                     'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.1',
                     'mistral-46b': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
                     }
float16_models = ['gpt-j-6B', 'gpt-neox-20b', 'llama-13b', 'bloom-7b1', 'opt-13b', 'llama3-70b', 'mistral-46b']


def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

def from_pretrained_tokenizer(cls, model_name, kwargs, cache_dir, use_ft=False):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)

    if use_ft:
        return cls.from_pretrained(cache_dir, **kwargs)
    else:
        return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

def load_model(args, model_name, device, cache_dir, use_ft=False, data_generation_flag=False):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
        

    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    if data_generation_flag:
        model_kwargs.update(dict(device_map='balanced')) #balanced_low_0, balanced

    model_kwargs.update(dict(quantization_config=bnb_config))
    model_kwargs.update(dict(torch_dtype=torch.bfloat16))


    if use_ft:
        model = AutoPeftModelForCausalLM.from_pretrained(cache_dir, **model_kwargs, cache_dir=args.cache_dir)
    else:
        model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    if not model.device.type=='cuda':
        model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name, for_dataset, cache_dir, use_ft, scoring_model=False):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained_tokenizer(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir, use_ft=use_ft)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer

import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)

from mix_model.dexperts import DExpertsLlama
def load_dexperts_model_and_tokenizer(
    large_model,
    small_model,
    small_finetune_model,
    tokenizer,
    alpha: float = 1.0,
    max_seq_len: int=384,
    mix_ratio: float=1.0
):
    model = DExpertsLlama(
        large_model=large_model,
        small_model=small_model,
        small_finetune_model=small_finetune_model,
        tokenizer=tokenizer,
        alpha=alpha,
        max_seq_len=max_seq_len,
        mix_ratio=mix_ratio)

    return model, tokenizer

