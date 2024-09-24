import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pickle as pkl

chat=True

def eval_data_prepare(args):
    if args.ft_detection_method in ["fast-detectgpt"]:
        if args.use_mix:
            prompt_sample_eval_path = f"{args.dataset_file}/gen_data/mixft_{args.base_model_name}/prompt_gen_eval_mix_{args.mix_ratio}_beta_{args.dpo_beta}_ep_{args.dpo_epoch}_scoring_{args.ft_scoring_model_name}_reference_{args.ft_reference_model_name}.json"
        else:
            prompt_sample_eval_path = f"{args.dataset_file}/gen_data/gen_{args.base_model_name}/prompt_gen_eval.json"
    if args.ft_detection_method in ["roberta-base", "roberta-large"]:
        if args.use_mix:
            prompt_sample_eval_path = f"{args.dataset_file}/gen_data/mixft_{args.base_model_name}/prompt_gen_eval_{args.ft_detection_method}_mix_{args.mix_ratio}_beta_{args.dpo_beta}_ep_{args.dpo_epoch}.json"
        else:
            prompt_sample_eval_path = f"{args.dataset_file}/gen_data/gen_{args.base_model_name}/prompt_gen_eval.json"

    with open(prompt_sample_eval_path, 'rb') as f:
        data_unprocessed = json.load(f)

    full_eval_file=f"{args.dataset_file}/full_texts/full_eval_texts.pkl"
    with open(full_eval_file, 'rb') as f:
        full_eval = pkl.load(f)
    data = {}
    if chat:
        p = 'prompt'
        s = 'sampled'
        r= 'real'
        data[p] = [item.split('[/INST]')[0].lstrip() for item in data_unprocessed[p]]
        data[s] = [item.split('[/INST]')[0].lstrip() for item in data_unprocessed[s]]
        data[r] =[item for item in full_eval]
    else:
        data = data_unprocessed
        r = 'real'
        data[r] = [item for item in full_eval]


    eval_prompts = data['prompt'][0:args.detect_num]
    eval_sampled = data['sampled'][0:args.detect_num]
    eval_real = data['real'][0:args.detect_num]
    eval_data = {}
    eval_data['prompt']=eval_prompts
    eval_data['sampled'] = eval_sampled
    eval_data['real'] = eval_real

    return eval_data