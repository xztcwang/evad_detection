import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import numpy as np
import torch
import tqdm
import argparse
import json
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from utils.metrics import  get_roc_metrics,get_precision_recall_metrics
from utils.model_utils import get_model_fullname, from_pretrained
from detection_eval.eval_utils import eval_data_prepare
from dpo_builders.detectors_utils.detectgpt import DetectGPT
from utils.load_save_file_utils import setup_seed

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

chat=True
use_LABEL=True

def load_mask_model(model_name, device, cache_dir):
    model_name = get_model_fullname(model_name)
    # mask filling t5 model
    print(f'Loading mask filling model {model_name}...')
    model_kwargs = {}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_kwargs.update(dict(device_map='balanced'))
    model_kwargs.update(dict(quantization_config=bnb_config))
    model_kwargs.update(dict(torch_dtype=torch.bfloat16))
    mask_model = from_pretrained(AutoModelForSeq2SeqLM, model_name, model_kwargs, cache_dir)
    return mask_model

def load_mask_tokenizer(model_name, max_length, cache_dir):
    model_name = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    optional_tok_kwargs['model_max_length'] = max_length
    tokenizer = from_pretrained(AutoTokenizer, model_name, optional_tok_kwargs, cache_dir)
    return tokenizer

def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    buffer_size = 1
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

# replace each masked span with a sample from T5 mask_model
def replace_masks(args, mask_model, mask_tokenizer, texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(args.device)
    outputs = mask_model.generate(**tokens, max_length=args.max_length, do_sample=True, top_p=args.mask_top_p,
                                  num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

def perturb_texts_(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    span_length = args.span_length
    pct = args.pct_words_masked
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
        if attempts>10:
            setup_seed(args.seed+attempts)
    return perturbed_texts

def perturb_texts(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    chunk_size = 10
    outputs = []
    for i in range(0, len(texts), chunk_size):
        perturbed_texts=perturb_texts_(args, mask_model, mask_tokenizer, texts[i:i + chunk_size], ceil_pct=ceil_pct)
        outputs.extend(perturbed_texts)
    return outputs

def load_pert_data(input_file):
    with open(input_file, "r") as fin:
        data = json.load(fin)
    return data

def save_pert_data(output_file, data):
    with open(output_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"perturb data written ")

def generate_perturbs(args, perturb_file):
    n_perturbations = args.n_perturbations
    # load model
    mask_model = load_mask_model(args.mask_filling_model_name, args.device, args.cache_dir)
    mask_model.eval()
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = load_mask_tokenizer(args.mask_filling_model_name, n_positions, args.cache_dir)
    eval_data=eval_data_prepare(args)

    n_samples = len(eval_data["sampled"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # generate perturb samples

    perturbs = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Perturb text"):
        real_text = eval_data["real"][idx]
        sampled_text = eval_data["sampled"][idx]

        # perturb
        p_sampled_text= perturb_texts(args, mask_model, mask_tokenizer, [sampled_text for _ in range(n_perturbations)])
        p_real_text= perturb_texts(args, mask_model, mask_tokenizer,
                                        [real_text for _ in range(n_perturbations)])
        assert len(
            p_sampled_text) == n_perturbations, f"Expected {n_perturbations} perturbed samples, got {len(p_sampled_text)}"
        assert len(
            p_real_text) == n_perturbations, f"Expected {n_perturbations} perturbed samples, got {len(p_real_text)}"
        # result
        perturbs.append({
            "real": real_text,
            "sampled": sampled_text,
            "perturbed_sampled": p_sampled_text,
            "perturbed_real": p_real_text
        })

    save_pert_data(perturb_file, perturbs)


def detect_texts(args):

    n_perturbations = args.n_perturbations
    name = f'perturbation_{n_perturbations}'

    if args.use_mix:
        perturb_file = f'{args.dataset_file}/gen_data/mixft_{args.base_model_name}/mix_{args.mix_ratio}_perturbing_{args.mask_filling_model_name}_scoring_{args.scoring_model_name}_pctwordsmasked_{args.pct_words_masked}.{name}.beta_{args.dpo_beta}_ep_{args.dpo_epoch}.json'
    else:
        perturb_file = f'{args.dataset_file}/gen_data/gen_{args.base_model_name}/perturbing_{args.mask_filling_model_name}_scoring_{args.scoring_model_name}_pctwordsmasked_{args.pct_words_masked}.{name}.json'

    if os.path.exists(perturb_file):
        print(f'Use existing perturbation file: {perturb_file}')
    else:
        generate_perturbs(args, perturb_file)

    detector=DetectGPT(args)
    pert_eval_data=load_pert_data(perturb_file)
    eval_preds=detector(pert_eval_data)
    predictions = {'real': [x["real_crit"] for x in eval_preds],
                   'sampled': [x["sampled_crit"] for x in eval_preds]}
    text_pairs = {'real': [x["real"] for x in eval_preds],
                  'sampled': [x["sampled"] for x in eval_preds]}
    return predictions, text_pairs

def detect_eval(predictions,text_pairs):
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['sampled'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['sampled'])
    pred_results = {'name': f'evaluation by {args.detection_method} on {args.dataset}',
                    'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                    'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                    'detect_preds': predictions
                    }
    result_file=f"{args.results_file}/{args.detection_method}"

    if args.use_mix:
        results_path = f"{result_file}/mixft_{args.base_model_name}_eval_beta_{args.dpo_beta}_ep_{args.dpo_epoch}_scoring_{args.scoring_model_name}_mask_{args.mask_filling_model_name}.json"
    else:
        results_path = f"{result_file}/{args.base_model_name}_eval_scoring_{args.scoring_model_name}_mask_{args.mask_filling_model_name}.json"

    with open(results_path, 'w') as fout:
        json.dump(pred_results, fout)
    return results_path

def open_results(results_path):
    with open(results_path, 'r') as fin:
        res = json.load(fin)
        return res
def get_results(results_path):
    res=open_results(results_path)

    def get_auroc_fpr_tpr(res):
        return res['metrics']['roc_auc'], res['metrics']['fpr'], res['metrics']['tpr']

    def get_pr_metrics(res):
        return res['pr_metrics']['pr_auc'], res['pr_metrics']['precision'], res['pr_metrics']['recall']

    from statistics import mean
    cols=get_auroc_fpr_tpr(res)
    print("ROC_AUC: {:.4f}, FPR: {:.4f}, TPR: {:.4f}".format(cols[0], mean(cols[1]), mean(cols[2])))

    cols=get_pr_metrics(res)
    print("PR_AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(cols[0],  mean(cols[1]),  mean(cols[2])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_method', type=str, default="detectgpt")
    parser.add_argument('--ft_detection_method', type=str, default="fast-detectgpt")
    parser.add_argument('--dataset', type=str, default="openwebtext")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--base_model_name', type=str, default="llama2-13b")#mistral-46b, llama3-70b, llama2-13b
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-3b")
    parser.add_argument('--pct_words_masked', type=float,
                        default=0.3)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_perturbations', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")#gpt-neo-2.7B, llama2-7b
    parser.add_argument('--cache_dir', type=str,
                        default="../../../detectionfiles/cache")
    parser.add_argument('--dataset_file', type=str,
                        default="../../../detectionfiles/openwebtext")
    parser.add_argument('--results_file', type=str,
                        default="../../../detectionfiles/results/openwebtext")
    parser.add_argument('--dpo_beta', type=float, default=0.1)
    parser.add_argument('--dpo_epoch', type=int, default=5)
    parser.add_argument('--use_for_score', action='store_true')
    parser.add_argument('--use_ft', action='store_false')
    parser.add_argument('--use_mix', action='store_false')
    parser.add_argument('--mix_ratio', type=float, default=0.5)
    parser.add_argument('--ft_reference_model_name', type=str, default="llama2-7b")  # mistral-7b, gpt-j-6B, llama2-7b, llama3-8b
    parser.add_argument('--ft_scoring_model_name', type=str, default="llama2-7b")  # mistral-7b, gpt-neo-2.7B, llama2-7b
    parser.add_argument('--detect_num', type=int, default=500)
    args = parser.parse_args()
    setup_seed(args.seed)
    predictions, text_pairs = detect_texts(args)
    results_path=detect_eval(predictions, text_pairs)
    get_results(results_path)


