import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import json
from utils.model_utils import load_model,load_tokenizer
from utils.load_save_file_utils import setup_seed
from dpo_builders.detectors_utils.baseline_detector import Baseline_Detectors
from dpo_builders.detectors_utils.fast_detectgpt import FastDetectGPT
from dpo_builders.detectors_utils.roberta_lg import Commercial_Detectors
from dpo_builders.detectors_utils.detectgpt import DetectGPT
from dpo_builders.detectors_utils.detect_llm import DetectLLM
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

SEPARATOR = '<<<SEP>>>'
chat=True
start_idx = 0
end_idx = None
reverse=False


def load_gen_train_data(file_path, args):
    gen_file_path=f"{file_path}/gen_data/gen_{args.base_model_name}/prompt_gen_train.json"
    with open(gen_file_path, 'rb') as f:
        data_unprocessed = json.load(f)
    if chat:
        s1 = 'sampled1'
        s2 = 'sampled2'
        data = {}
        data[s1] = [item.split('[/INST]')[0].lstrip() for item in data_unprocessed[s1]]
        data[s2] = [item.split('[/INST]')[0].lstrip() for item in data_unprocessed[s2]]
    else:
        data = data_unprocessed
    return data

def load_perturb_file(args):
    n_perturbations = args.n_perturbations
    name = f'perturbation_{n_perturbations}'
    if args.use_mix:
        perturb_file = f'{args.dataset_file}/gen_data/mix_{args.base_model_name}/perturbing_{args.mask_filling_model_name}_scoring_{args.scoring_model_name}.{name}.json'
    else:
        perturb_file = f'{args.dataset_file}/gen_data/gen_{args.base_model_name}/perturbing_{args.mask_filling_model_name}_scoring_{args.scoring_model_name}.{name}.json'
    with open(perturb_file, "r") as fin:
        data = json.load(fin)
    return data


class DPO_MixBuilder:
    def __init__(self, args):
        self.args = args
        self.base_tokenizer = load_tokenizer(args.base_model_name, args.dataset, args.cache_dir, use_ft=False)
        self.base_model = load_model(args, args.base_model_name, args.device,
                                      args.cache_dir, use_ft=False,data_generation_flag=False)
        self.base_model.resize_token_embeddings(len(self.base_tokenizer))
        self.detection_list=[args.detection_method]

        for detector_name in self.detection_list:
            self.detector_name=detector_name
            if detector_name in ["Likelihood", "Entropy", "LogRank", "Rank"]:
                self.detector = Baseline_Detectors(args=args)
            if detector_name == "detectgpt":
                self.detector=DetectGPT(args=args)
            if detector_name == "detectllm":
                self.detector = DetectLLM(args=args)
            if detector_name=="fast-detectgpt":
                self.detector = FastDetectGPT(args=args)
            if detector_name in ["roberta-large", "roberta-base"]:
                self.detector = Commercial_Detectors(args=args)

    def get_all_samples_probs(self, detector_name, two_sampled_data, perturb_file=None, raw_data=None):

        if detector_name in ["detectgpt","detectllm"]:
            probs_file_path = f"{self.args.dataset_file}/gen_data/{args.base_model_name}_{detector_name}_probs_gen_train.json"
            ai_like_results1 = self.detector(data=perturb_file, sampled_data=two_sampled_data['sampled1'])
            probs1=ai_like_results1["sampled_ailike"]
            ai_like_results2 = self.detector(data=perturb_file, sampled_data=two_sampled_data['sampled2'])
            probs2 = ai_like_results2["sampled_ailike"]
            dump = {'probs 1': probs1, 'probs 2': probs2}
            with open(probs_file_path, 'w') as f:
                json.dump(dump, f)
        if detector_name in ["Likelihood", "Entropy", "LogRank", "Rank"]:
            probs_file_path = f"{self.args.dataset_file}/gen_data/{args.base_model_name}_{detector_name}_probs_gen_train.json"
            probs1 = self.detector(data=two_sampled_data['sampled1'])
            probs2 = self.detector(data=two_sampled_data['sampled2'])

            dump = {'probs 1': probs1, 'probs 2': probs2}
            with open(probs_file_path, 'w') as f:
                json.dump(dump, f)
        if detector_name in ["fast-detectgpt"]:
            probs_file_path = f"{self.args.dataset_file}/gen_data/{args.base_model_name}_{detector_name}_scoring.{self.args.scoring_model_name}_reference.{self.args.reference_model_name}_probs_gen_train.json"
            probs1 = self.detector(data=two_sampled_data['sampled1'])
            probs2 = self.detector(data=two_sampled_data['sampled2'])
            dump = {'probs 1': probs1, 'probs 2': probs2}
            with open(probs_file_path, 'w') as f:
                json.dump(dump, f)
        if detector_name in ["roberta-large", "roberta-base"]:
            probs_file_path = f"{self.args.dataset_file}/gen_data/{args.base_model_name}_{detector_name}_probs_gen_train.json"
            probs1 = self.detector(data=two_sampled_data['sampled1'])
            probs2 = self.detector(data=two_sampled_data['sampled2'])
            dump = {'probs 1': probs1, 'probs 2': probs2}
            with open(probs_file_path, 'w') as f:
                json.dump(dump, f)

    def logic_chosen_rejected(self,gen_data, gen_probs):
        prompt = []
        chosen = []
        rejected = []
        if self.detector_name in ["roberta-large", "roberta-base"]:
            for i in range(len(gen_probs['probs 1'])):
                prompt_text = gen_data['prompt'][i]
                less = (gen_probs['probs 1'][i] < gen_probs['probs 2'][i])
                more = (gen_probs['probs 1'][i] > gen_probs['probs 2'][i])
                if (more and not reverse) or (less and reverse):
                    chosen.append(gen_data['sampled1'][i])
                    rejected.append(gen_data['sampled2'][i])
                elif (less and not reverse) or (more and reverse):
                    chosen.append(gen_data['sampled2'][i])
                    rejected.append(gen_data['sampled1'][i])
                else:
                    continue
                prompt.append(prompt_text)
        if self.detector_name in ["detectgpt", "detectllm", "fast-detectgpt", "Likelihood", "LogRank"]:
            for i in range(len(gen_probs['probs 1'])):
                prompt_text = gen_data['prompt'][i]
                less = (gen_probs['probs 1'][i] < gen_probs['probs 2'][i])
                more = (gen_probs['probs 1'][i] > gen_probs['probs 2'][i])
                if (less and not reverse) or (more and reverse):
                    chosen.append(gen_data['sampled1'][i])
                    rejected.append(gen_data['sampled2'][i])
                elif (more and not reverse) or (less and reverse):
                    chosen.append(gen_data['sampled2'][i])
                    rejected.append(gen_data['sampled1'][i])
                else:
                    continue
                prompt.append(prompt_text)

        return prompt,chosen, rejected

    def save_dpo_pf_data(self,detector_name):
        if detector_name in ["Likelihood", "Entropy", "LogRank", "Rank"]:
            probs_file_path = f"{self.args.dataset_file}/gen_data/{args.base_model_name}_{detector_name}_probs_gen_train.json"
        if detector_name in ["fast-detectgpt"]:
            probs_file_path = f"{self.args.dataset_file}/gen_data/{args.base_model_name}_{detector_name}_scoring.{self.args.scoring_model_name}_reference.{self.args.reference_model_name}_probs_gen_train.json"
        if detector_name in ["roberta-large", "roberta-base"]:
            probs_file_path = f"{self.args.dataset_file}/gen_data/{args.base_model_name}_{detector_name}_probs_gen_train.json"
        with open(probs_file_path, 'rb') as f:
            gen_probs = json.load(f)
        gen_file_path = f"{self.args.dataset_file}/gen_data/gen_{self.args.base_model_name}/prompt_gen_train.json"
        with open(gen_file_path, 'rb') as f:
            gen_data = json.load(f)
        prompt,chosen, rejected=self.logic_chosen_rejected(gen_data, gen_probs, detector_name)
        if detector_name in ["fast-detectgpt"]:
            dpo_file_path = f"{self.args.dataset_file}/dpo_data/{self.args.base_model_name}.{detector_name}_scoring.{self.args.scoring_model_name}_reference.{self.args.reference_model_name}_dpo_train.json"
        if detector_name in ["roberta-large", "roberta-base"]:
            dpo_file_path = f"{self.args.dataset_file}/dpo_data/{self.args.base_model_name}.{detector_name}_dpo_train.json"
        with open(dpo_file_path, 'w') as f:
            json.dump({'prompt': prompt, 'chosen': chosen, 'rejected': rejected}, f, indent=2)


def experiment(args):
    setup_seed(args.seed)
    prompt_gen_train_data = load_gen_train_data(args.dataset_file, args)
    dpo_mixbuilder = DPO_MixBuilder(args)
    detector=args.detection_method
    if detector in ["detectgpt", "detectllm"]:
        perturb_file = load_perturb_file(args)
        dpo_mixbuilder.get_all_samples_probs(detector_name=detector,
                                       two_sampled_data=prompt_gen_train_data,
                                       perturb_file=perturb_file)
        dpo_mixbuilder.save_dpo_pf_data(detector)
    if detector in ["Likelihood", "Entropy", "LogRank", "Rank", "fast-detectgpt", "roberta-base", "roberta-large"]:
        dpo_mixbuilder.get_all_samples_probs(detector_name=detector,
                                                 two_sampled_data=prompt_gen_train_data,
                                                 perturb_file=None, raw_data=None)
        dpo_mixbuilder.save_dpo_pf_data(detector)
    print("complete")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name', type=str, default="llama2-7b")
    parser.add_argument('--reference_model_name', type=str, default="llama2-7b")  # gpt-j-6B, llama2-7b
    parser.add_argument('--scoring_model_name', type=str, default="llama2-7b")  # gpt-neo-2.7B, llama2-7b
    parser.add_argument('--detection_method', type=str, default="fast-detectgpt") #roberta-large, fast-detectgpt

    parser.add_argument('--dataset', type=str, default="openwebtext")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str,
                        default="../../../../detectionfiles/cache")
    parser.add_argument('--dataset_file', type=str,
                        default="../../../../detectionfiles/openwebtext")
    parser.add_argument('--discrepancy_analytic', action='store_true')
    parser.add_argument('--n_perturbations', type=int, default=100)
    parser.add_argument('--dpo_beta', type=float, default=0.1)
    parser.add_argument('--dpo_epoch', type=int, default=5)
    parser.add_argument('--m_length', type=int, default=200)
    parser.add_argument('--detect_batch_size', type=int, default=20)

    parser.add_argument('--use_ft', action='store_true')
    parser.add_argument('--use_mix', action='store_true')
    parser.add_argument('--use_for_score', action='store_false')

    parser.add_argument('--sample_batch_size', type=int, default=200)
    parser.add_argument('--start_idx', type=int, default=0)
    args = parser.parse_args()

    experiment(args)