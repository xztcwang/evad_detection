from utils.model_utils import load_model,load_tokenizer
import torch
import tqdm
import numpy as np


class DetectGPT:
    def __init__(self, args):
        self.args=args
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir, use_ft=False, scoring_model=True)
        self.scoring_model = load_model(args, args.scoring_model_name, args.device,
                                   args.cache_dir, use_ft=False, data_generation_flag=True)
        self.scoring_model.eval()

        self.pert_name = f'perturbation_{args.n_perturbations}'


    # Get the log likelihood of each text under the base_model
    def get_ll(self, args, scoring_model, scoring_tokenizer, text):
        with torch.no_grad():
            tokenized = scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids
            return -scoring_model(**tokenized, labels=labels).loss.item()

    def get_lls(self, args, scoring_model, scoring_tokenizer, texts):
        return [self.get_ll(args, scoring_model, scoring_tokenizer, text) for text in texts]

    def __call__(self, data, sampled_data=None):
        if self.args.use_for_score:
            results = data
            n_samples = len(data)
            for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {self.pert_name} criterion"):
                sampled_text=sampled_data[idx]
                perturbed_sampled = results[idx]["perturbed_sampled"]
                # sampled text
                sampled_ll = self.get_ll(self.args, self.scoring_model, self.scoring_tokenizer, sampled_text)
                p_sampled_ll = self.get_lls(self.args, self.scoring_model, self.scoring_tokenizer, perturbed_sampled)
                # result
                results[idx]["sampled_ll"] = sampled_ll
                results[idx]["all_perturbed_sampled_ll"] = p_sampled_ll
                results[idx]["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
                results[idx]["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1

            # compute diffs with perturbed
            predictions = {'sampled': []}
            for res in results:
                if res['perturbed_sampled_ll_std'] == 0:
                    res['perturbed_sampled_ll_std'] = 1
                    print("WARNING: std of perturbed sampled is 0, setting to 1")
                    print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                predictions['sampled'].append(
                    (res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])
            final_results = []
            for idx in tqdm.tqdm(range(n_samples), desc=f"Collecting Results"):
                sampled_crit = predictions['sampled'][idx]
                final_results.append(sampled_crit)
            return final_results
        else:
            results = data
            n_samples = len(data)
            for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {self.pert_name} criterion"):
                real_text = results[idx]["real"]
                sampled_text = results[idx]["sampled"]
                perturbed_real = results[idx]["perturbed_real"]
                perturbed_sampled = results[idx]["perturbed_sampled"]
                # real text
                real_ll = self.get_ll(self.args, self.scoring_model, self.scoring_tokenizer, real_text)
                p_real_ll = self.get_lls(self.args, self.scoring_model, self.scoring_tokenizer, perturbed_real)
                # sampled text
                sampled_ll = self.get_ll(self.args, self.scoring_model, self.scoring_tokenizer, sampled_text)
                p_sampled_ll = self.get_lls(self.args, self.scoring_model, self.scoring_tokenizer, perturbed_sampled)
                # result
                results[idx]["real_ll"] = real_ll
                results[idx]["sampled_ll"] = sampled_ll
                results[idx]["all_perturbed_sampled_ll"] = p_sampled_ll
                results[idx]["all_perturbed_real_ll"] = p_real_ll
                results[idx]["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
                results[idx]["perturbed_real_ll"] = np.mean(p_real_ll)
                results[idx]["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
                results[idx]["perturbed_real_ll_std"] = np.std(p_real_ll) if len(p_real_ll) > 1 else 1
            # compute diffs with perturbed
            predictions = {'real': [], 'sampled': []}
            for res in results:
                if res['perturbed_real_ll_std'] == 0:
                    res['perturbed_real_ll_std'] = 1
                    print("WARNING: std of perturbed real is 0, setting to 1")
                    print(f"Number of unique perturbed real texts: {len(set(res['perturbed_real']))}")
                    print(f"Real text: {res['real']}")
                if res['perturbed_sampled_ll_std'] == 0:
                    res['perturbed_sampled_ll_std'] = 1
                    print("WARNING: std of perturbed sampled is 0, setting to 1")
                    print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                    print(f"Sampled text: {res['sampled']}")
                predictions['real'].append(
                        (res['real_ll'] - res['perturbed_real_ll']) / res['perturbed_real_ll_std'])
                predictions['sampled'].append(
                        (res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])
            final_results = []
            for idx in tqdm.tqdm(range(n_samples), desc=f"Collecting Results"):
                real_text = results[idx]["real"]
                sampled_text = results[idx]["sampled"]
                real_crit = predictions['real'][idx]
                sampled_crit = predictions['sampled'][idx]
                # result
                final_results.append({"real": real_text,
                                      "real_crit": real_crit,
                                      "sampled": sampled_text,
                                      "sampled_crit": sampled_crit})
            return final_results
