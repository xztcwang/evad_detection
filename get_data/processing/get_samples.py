import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import random
import json
import tqdm
import argparse
import pickle as pkl
from utils.model_utils import load_model,load_tokenizer

def get_samples(args, model, tokenizer, prompts, seed1=0, seed2=1, min_length=150,min_words=55):
    inputs = tokenizer(prompts, padding=True, return_tensors='pt', return_token_type_ids=False).to(args.device)


    tries = 0
    m = 0
    while m < min_words:
        if tries != 0:
            print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
        random_int = random.randint(1, 10)
        seed1=seed1*random_int
        torch.manual_seed(seed1)
        outputs1 = model.generate(
            **inputs,
            do_sample=True,
            max_length=args.m_length,
            min_length=min_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p
        )
        decoded1=tokenizer.batch_decode(outputs1, skip_special_tokens=True)
        m = min(len(x.split()) for x in decoded1)
        tries += 1

    tries = 0
    m = 0
    while m < min_words:
        if tries != 0:
            print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
        random_int = random.randint(1, 10)
        seed2 = seed2 * random_int
        torch.manual_seed(seed2)
        outputs2 = model.generate(
            **inputs,
            do_sample=True,
            max_length=args.m_length,
            min_length=min_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p
        )
        decoded2 = tokenizer.batch_decode(outputs2, skip_special_tokens=True)
        m = min(len(x.split()) for x in decoded2)
        tries += 1

    return decoded1, decoded2


def prepare_sampling_datasets(args):
    sampling_tokenizer = load_tokenizer(args.sample_model_name, args.dataset, args.cache_dir, use_ft=False)
    sampling_model = load_model(args, args.sample_model_name, args.device, args.cache_dir, use_ft=False,
                                data_generation_flag=True)
    sampling_model.eval()
    prompt_train_datapath = f"{args.dataset_file}/prompts_data/prompt_train_texts.pkl"
    print("Loading data.")
    with open(prompt_train_datapath, 'rb') as f:
        prompts_train = pkl.load(f)
    start_idx = 0
    end_idx = None
    if end_idx == None:
        end_idx = len(prompts_train)
    print('Overall Start:', start_idx)
    print('Overall End:', end_idx)
    gens1 = []
    gens2 = []
    print("Getting generations.")


    for j in tqdm.tqdm(list(range(start_idx, end_idx, args.batch_size))):
        batch = prompts_train[j:j + args.batch_size]
        out1, out2 = get_samples(args, sampling_model, sampling_tokenizer,
                                 batch, seed1=2 * j, seed2=2 * j + 1, min_length=150 if args.dataset in ['pubmed'] else 150, min_words=30 if args.dataset in ['pubmed'] else 55)
        gens1.extend(out1)
        gens2.extend(out2)
    gens1 = [g.replace('<unk>', '').replace('<s> ', '').replace('</s>', '').replace('<|eot_id|>', '').replace(
        '<|begin_of_text|>', '').replace('<|end_of_text|>', '').replace('<|start_header_id|>', '').replace(
        '<|end_header_id|>', '') for g in gens1]
    gens2 = [g.replace('<unk>', '').replace('<s> ', '').replace('</s>', '').replace('<|eot_id|>', '').replace(
        '<|begin_of_text|>', '').replace('<|end_of_text|>', '').replace('<|start_header_id|>', '').replace(
        '<|end_header_id|>', '') for g in gens2]
    gen_datafile = f"{args.dataset_file}/gen_data/gen_{args.sample_model_name}/prompt_gen_train.json"
    with open(gen_datafile, 'w') as f:
        json.dump({'prompt': prompts_train, 'sampled1': gens1, 'sampled2': gens2}, f, indent=2)
    print("complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_model_name', type=str, default="llama2-13b")
    parser.add_argument('--dataset', type=str, default="openwebtext")
    parser.add_argument('--cache_dir', type=str,
                        default="../../../../detectionfiles/cache")
    parser.add_argument('--dataset_file', type=str,
                        default="../../../../detectionfiles/openwebtext")
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--m_length', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    prepare_sampling_datasets(args)





