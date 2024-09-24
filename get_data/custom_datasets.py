import random
import datasets
import os

SEPARATOR = '<<<SEP>>>'

DATASETS = ['writing', 'english', 'german', 'pubmed']

def load_dataset(path, name=None, split=None, cache_dir=None):
    # use local model if it exists
    local_path = os.path.join(cache_dir, f'local.{path}_{name}_{split}')
    if os.path.exists(local_path):
        return datasets.load_from_disk(local_path)
    return datasets.load_dataset(path, name, split=split, cache_dir=cache_dir)

def load_pubmed(cache_dir):
    data_labeled = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    data_unlabeled = datasets.load_dataset('pubmed_qa', 'pqa_unlabeled', split='train', cache_dir=cache_dir)
    # combine question and long_answer
    data_labeled = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data_labeled['question'], data_labeled['long_answer'])]

    data_unlabeled = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data_unlabeled['question'], data_unlabeled['long_answer'])]
    data=data_labeled+data_unlabeled
    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(data_path=None):

    with open(f'{data_path}/valid.wp_source', 'r') as f:
        prompts = f.readlines()
    with open(f'{data_path}/valid.wp_target', 'r') as f:
        stories = f.readlines()

    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = datasets.load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        if name=="writing":
            return load_fn(data_path=cache_dir)
        elif name=="pubmed":
            return load_pubmed(cache_dir=cache_dir)
        else:
            return load_fn(cache_dir=cache_dir)
    else:
        raise ValueError(f'Unknown dataset {name}')