import pickle as pkl
import random
import argparse
import os
import get_data.custom_datasets as custom_datasets
from utils.load_save_file_utils import setup_seed
from utils.model_utils import load_model,load_tokenizer

dataset_keys = {'xsum': 'document', 'squad': 'context', 'writing': 'document'}

def prepare_dataset(args, dataset, key=None):
    # strip newlines from each example; replace one or more newlines with a single space
    def _strip_newlines(text):
        return ' '.join(text.split())
    # load data
    if dataset in custom_datasets.DATASETS:
        if dataset=="writing":
            data = custom_datasets.load(dataset, args.load_data_path)
        else:
            data = custom_datasets.load(dataset, args.cache_dir)
    else:
        data = custom_datasets.load_dataset(dataset, split='train', cache_dir=args.cache_dir)[key]

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [_strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    random.shuffle(data)
    sampling_tokenizer = load_tokenizer(args.sample_model_name, args.dataset, args.cache_dir, use_ft=False)
    sampling_model = load_model(args, args.sample_model_name, args.device, args.cache_dir, use_ft=False, data_generation_flag=True)
    sampling_model.eval()

    truncate_tokenizer = load_tokenizer(args.truncate_model_name, args.dataset, args.cache_dir, use_ft=False)
    truncate_model = load_model(args, args.truncate_model_name, args.device, args.cache_dir, use_ft=False,
                                data_generation_flag=True)
    truncate_model.eval()

    truncate_tokenized_data=truncate_tokenizer(data)
    data = [x for x, y in zip(data, truncate_tokenized_data["input_ids"]) if len(y) <= 512]
    print(len(data))


    # Truncate texts to first 8 GPT2 tokens
    if dataset =="pubmed":
        prompts = [t[:t.index(custom_datasets.SEPARATOR)] for t in data]
    else:
        tokens = sampling_tokenizer(data, truncation=True, max_length=1024)['input_ids']
        prefix_tokens = [t[:8] for t in tokens]
        prompts = sampling_tokenizer.batch_decode(prefix_tokens)


    prompts_eval_file = f"{args.dataset_file}/prompts_data/prompt_eval_texts.pkl"
    prompts_train_file = f"{args.dataset_file}/prompts_data/prompt_train_texts.pkl"
    full_eval_file = f"{args.dataset_file}/full_texts/full_eval_texts.pkl"
    full_train_file = f"{args.dataset_file}/full_texts/full_train_texts.pkl"
    with open (prompts_eval_file, 'wb') as f:
        pkl.dump(prompts[:500], f)
    with open (prompts_train_file, 'wb') as f:
        pkl.dump(prompts[500:10500], f)
    with open(full_eval_file, 'wb') as f:
        pkl.dump(data[:500], f)
    with open(full_train_file, 'wb') as f:
        pkl.dump(data[500:10500], f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="openwebtext")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--sample_model_name', type=str, default="gpt2-xl")
    parser.add_argument('--truncate_model_name', type=str, default="llama2-7b")
    parser.add_argument('--cache_dir', type=str,
                        default="../../../../detectionfiles/gpt2_cache")
    parser.add_argument('--dataset_file', type=str,
                        default="../../../../detectionfiles/openwebtext")
    parser.add_argument('--load_data_path', type=str,
                        default="../../../../detectionfiles/writing/writingPrompts")
    args = parser.parse_args()

    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    setup_seed(args.seed)
    if args.dataset in ['xsum', 'squad', 'writing']:
        prepare_dataset(args, args.dataset, dataset_keys[args.dataset])
    else:
        prepare_dataset(args, args.dataset)
    print("complete")

