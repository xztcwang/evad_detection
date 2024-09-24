from utils.model_utils import load_model,load_tokenizer
import torch
import tqdm
import numpy as np

class DetectLLM:
    def __init__(self, args):
        self.args=args
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir, use_ft=False, scoring_model=True)
        self.scoring_model = load_model(args, args.scoring_model_name, args.device,
                                        args.cache_dir, use_ft=False, data_generation_flag=True)
        self.scoring_model.eval()
        self.criterion_name=self.args.criterion_name

    def get_likelihood(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1

        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return log_likelihood.mean().item()

    def get_logrank(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        ranks = torch.log(ranks)
        return ranks.mean().item()

    # Log-Likelihood Log-Rank Ratio
    def get_lrr(self, text, perturbs):
        with torch.no_grad():
            tokenized = self.scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(self.args.device)
            labels = tokenized.input_ids[:, 1:]
            logits = self.scoring_model(**tokenized).logits[:, :-1]
            likelihood = self.get_likelihood(logits, labels)
            logrank = self.get_logrank(logits, labels)
            return - likelihood / logrank

    # Normalized Log-Rank Perturbation
    def get_npr(self, text, perturbs):
        with torch.no_grad():
            tokenized = self.scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(self.args.device)
            labels = tokenized.input_ids[:, 1:]
            logits = self.scoring_model(**tokenized).logits[:, :-1]
            logrank = self.get_logrank(logits, labels)
            # perturbations
            logranks = []
            for perturb in perturbs:
                tokenized = self.scoring_tokenizer(perturb, return_tensors="pt", return_token_type_ids=False).to(self.args.device)
                labels = tokenized.input_ids[:, 1:]
                logits = self.scoring_model(**tokenized).logits[:, :-1]
                logranks.append(self.get_logrank(logits, labels))
            # npr
            return np.mean(logranks) / logrank

    def __call__(self, data):
        results = data
        n_samples = len(data)
        # eval criterions
        if self.criterion_name=='lrr':
            self.criterion_fn=self.get_lrr
        if self.criterion_name=='npr':
            self.criterion_fn = self.get_npr

        final_results = []

        for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {self.criterion_name} criterion"):
            real_text = results[idx]["real"]
            sampled_text = results[idx]["sampled"]
            perturbed_real = results[idx]["perturbed_real"]
            perturbed_sampled = results[idx]["perturbed_sampled"]

            real_crit = self.criterion_fn(real_text, perturbed_real)
            sampled_crit = self.criterion_fn(sampled_text, perturbed_sampled)

            final_results.append({"real": real_text,
                                  "real_crit": real_crit,
                                  "sampled": sampled_text,
                                  "sampled_crit": sampled_crit})
        return final_results






