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
from utils.load_save_file_utils import setup_seed
from utils.model_utils import load_model,load_tokenizer,load_dexperts_model_and_tokenizer
from utils.metrics import  rouge_bert_scorer
from statistics import mean
random.seed(0)

def get_samples(args, model, tokenizer, prompts, full_texts, min_words=55):
    tries = 0
    m = 0
    prompt_tokens=30
    while m < min_words:
        if tries != 0:
            print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
        if args.use_mix:

            inputs = [
                    tokenizer(prompt, padding=False, return_tensors='pt', return_token_type_ids=False).to(args.device)
                    for prompt in prompts]
            outputs = model.generate_with_ref(
                encoded_tokens=inputs,
                cache_path=args.cache_dir,
                max_gen_len=args.mix_max_gen_len,
                temperature=args.mix_temp,
                top_p=args.mix_top_p)
        else:
            inputs = tokenizer(prompts, padding=True, return_tensors='pt', return_token_type_ids=False).to(args.device)
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_length=args.m_length,
                min_length=args.min_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=args.temperature,
                top_p=args.top_p
            )
        decoded=tokenizer.batch_decode(outputs)
        m = min(len(x.split()) for x in decoded)
        tries += 1

    return decoded


def get_bert_score(res):
    bscore_p=res['b_scores']['bscore_p']
    bscore_r = res['b_scores']['bscore_r']
    bscore_f1 = res['b_scores']['bscore_f1']
    return mean(bscore_p), mean(bscore_r), mean(bscore_f1)

def get_rouge_score(res):
    rouge_1=res['rouges']['rouge_1']
    rouge_2=res['rouges']['rouge_2']
    rouge_L=res['rouges']['rouge_L']
    return mean(rouge_1), mean(rouge_2), mean(rouge_L)


def prepare_eval_datasets(args):
    if args.use_mix:
        large_tokenizer = load_tokenizer(args.base_model_name, args.dataset, args.cache_dir, use_ft=False)
        large_model = load_model(args, args.base_model_name, args.device,
                                    args.cache_dir, use_ft=False, data_generation_flag=True)

        dpo_dir = os.path.join(args.dpo_dir, args.small_model_name)
        dpo_dir = os.path.join(dpo_dir, args.dataset)
        if args.ft_detection_method in ["fast-detectgpt"]:
            dpo_savepath = f"{args.ft_detection_method}_scoring_{args.scoring_model_name}_reference_{args.reference_model_name}"
        if args.ft_detection_method in ["roberta-large", "roberta-base"]:
            dpo_savepath = f"{args.ft_detection_method}"
        dpo_dir = os.path.join(dpo_dir, dpo_savepath)
        dpo_param_savepath = f"beta_{args.dpo_beta}_ep_{args.dpo_epoch}"
        dpo_dir = os.path.join(dpo_dir, dpo_param_savepath)
        ft_small_model=load_model(args, args.small_model_name, args.device,
                                        dpo_dir, use_ft=True, data_generation_flag=True)
        small_model = load_model(args, args.small_model_name, args.device,
                                     args.cache_dir, use_ft=False, data_generation_flag=True)
        base_model, base_tokenizer = load_dexperts_model_and_tokenizer(
                large_model=large_model,
                small_model=small_model,
                small_finetune_model=ft_small_model,
                tokenizer=large_tokenizer,
                alpha=1.0,
                max_seq_len=args.m_length,
                mix_ratio=args.mix_ratio)
    else:
        base_model = load_model(args, args.base_model_name, args.device,
                                args.cache_dir, args.use_ft, data_generation_flag=True)
        base_tokenizer = load_tokenizer(args.base_model_name, args.dataset, args.cache_dir, args.use_ft)
    prompt_eval_datapath = f"{args.dataset_file}/prompts_data/prompt_eval_texts.pkl"
    full_eval_file = f"{args.dataset_file}/full_texts/full_eval_texts.pkl"
    print("Loading data.")
    with open(prompt_eval_datapath, 'rb') as f:
        prompts_eval = pkl.load(f)
    with open(full_eval_file, 'rb') as f:
        full_eval = pkl.load(f)

    start_idx = 0
    end_idx = None
    if end_idx == None:
        end_idx = args.detect_num
    print('Overall Start:', start_idx)
    print('Overall End:', end_idx)
    gens = []
    print("Getting generations.")

    for j in tqdm.tqdm(list(range(start_idx, end_idx, args.batch_size))):

        batch_prompts = prompts_eval[j:j + args.batch_size]
        batch_full=full_eval[j:j + args.batch_size]

        out= get_samples(args, base_model, base_tokenizer,
                                 batch_prompts, batch_full, min_words=50)

        gens.extend(out)

    gens = [g.replace('<unk>', '').replace('<s>', '').replace('</s>', '').replace('<|eot_id|>', '').replace(
        '<|begin_of_text|>', '').replace('<|end_of_text|>', '').replace('<|start_header_id|>', '').replace(
        '<|end_header_id|>', '') for g in gens]
    rouge_1, rouge_2, rouge_L, BScore_P, BScore_R, BScore_F1 \
        = rouge_bert_scorer(full_eval, gens)
    gen_results = {'b_scores': {'bscore_p': BScore_P, 'bscore_r': BScore_R, 'bscore_f1': BScore_F1},
               'rouges': {'rouge_1': rouge_1, 'rouge_2': rouge_2, 'rouge_L': rouge_L}
               }
    cols = get_bert_score(gen_results)
    print("Bert_Score: F1={:.4f}, P={:.4f}, R={:.4f}".format(cols[0], cols[1], cols[2]))

    cols = get_rouge_score(gen_results)
    print("Rouge_Score: Rouge1={:.4f}, Rouge2={:.4f}, RougeL={:.4f}".format(cols[0], cols[1], cols[2]))

    if args.ft_detection_method in ["fast-detectgpt"]:
        if args.use_mix:
            gen_datafile = f"{args.dataset_file}/gen_data/mixft_{args.base_model_name}/prompt_gen_eval_mix_{args.mix_ratio}_beta_{args.dpo_beta}_ep_{args.dpo_epoch}_scoring_{args.scoring_model_name}_reference_{args.reference_model_name}.json"
            gen_resultsfile = f"{args.dataset_file}/gen_data/mixft_{args.base_model_name}/genresults_mix_{args.mix_ratio}_beta_{args.dpo_beta}_ep_{args.dpo_epoch}_scoring_{args.scoring_model_name}_reference_{args.reference_model_name}.json"
        else:
            gen_datafile = f"{args.dataset_file}/gen_data/gen_{args.base_model_name}/prompt_gen_eval.json"
            gen_resultsfile = f"{args.dataset_file}/gen_data/gen_{args.base_model_name}/genresults.json"
    if args.ft_detection_method in ["roberta-large", "roberta-base"]:
        if args.use_mix:
            gen_datafile = f"{args.dataset_file}/gen_data/mixft_{args.base_model_name}/prompt_gen_eval_{args.ft_detection_method}_mix_{args.mix_ratio}_beta_{args.dpo_beta}_ep_{args.dpo_epoch}.json"
            gen_resultsfile = f"{args.dataset_file}/gen_data/mixft_{args.base_model_name}/genresults_{args.ft_detection_method}_mix_{args.mix_ratio}_beta_{args.dpo_beta}_ep_{args.dpo_epoch}.json"
        else:
            gen_datafile = f"{args.dataset_file}/gen_data/gen_{args.base_model_name}/prompt_gen_eval.json"
            gen_resultsfile = f"{args.dataset_file}/gen_data/gen_{args.base_model_name}/genresults.json"


    with open(gen_datafile, 'w') as f:
        json.dump({'prompt': prompts_eval, 'sampled': gens}, f, indent=2)
    with open(gen_resultsfile, 'w') as fout:
        json.dump(gen_results, fout)
    print("complete")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_detection_method', type=str, default="fast-detectgpt") #roberta-large, fast-detectgpt
    parser.add_argument('--base_model_name', type=str, default="llama2-13b") #mistral-46b, llama3-70b
    parser.add_argument('--small_model_name', type=str, default="llama2-7b") #mistral-7b, llama3-8b
    parser.add_argument('--dataset', type=str, default="openwebtext")
    parser.add_argument('--dataset_file', type=str,
                        default="../../../../detectionfiles/openwebtext")
    parser.add_argument('--cache_dir', type=str,
                        default="../../../../detectionfiles/cache")
    parser.add_argument('--dpo_dir', type=str,
                        default="../../../../detectionfiles/dpo_checkpoint")
    parser.add_argument('--reference_model_name', type=str, default="llama2-7b")  # gpt-j-6B, llama2-7b
    parser.add_argument('--scoring_model_name', type=str, default="llama2-7b")  # gpt-neo-2.7B, llama2-7b
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--min_length', type=int, default=150)
    parser.add_argument('--m_length', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--mix_ratio', type=float, default=0.5)
    parser.add_argument('--use_ft', action='store_false')
    parser.add_argument('--use_mix', action="store_false")
    parser.add_argument('--dpo_beta', type=float, default=0.1)
    parser.add_argument('--dpo_epoch', type=int, default=5)
    parser.add_argument('--mix_temp', type=float, default=1.0)
    parser.add_argument('--mix_top_p', type=float, default=1.0)
    parser.add_argument('--mix_max_gen_len', type=int, default=200)
    parser.add_argument('--detect_num', type=int, default=500)
    args = parser.parse_args()

    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    print(f"Using cache dir {args.cache_dir}")
    setup_seed(args.seed)
    prepare_eval_datasets(args)