from utils.model_utils import load_model,load_tokenizer
import torch
import tqdm



class FastDetectGPT:
    def __init__(self, args):
        self.args=args
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir, use_ft=False, scoring_model=True)
        self.scoring_model = load_model(args, args.scoring_model_name, args.device,
                                   args.cache_dir, use_ft=False, data_generation_flag=True)
        self.scoring_model.eval()
        if args.reference_model_name != args.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset,
                                                      args.cache_dir, use_ft=False, scoring_model=True)
            self.reference_model = load_model(args, args.reference_model_name, args.device,
                                            args.cache_dir, use_ft=False, data_generation_flag=True)
            self.reference_model.eval()
        # evaluate criterion
        if self.args.discrepancy_analytic:
            self.name = "sampling_discrepancy_analytic"
            self.criterion_fn = self.get_sampling_discrepancy_analytic
        else:
            self.name = "sampling_discrepancy"
            self.criterion_fn = self.get_sampling_discrepancy

    def get_sampling_discrepancy_analytic(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        discrepancy = discrepancy.mean()
        return discrepancy.item()

    def get_sampling_discrepancy(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        samples = self.get_samples(logits_ref, labels)
        log_likelihood_x = self.get_likelihood(logits_score, labels)
        log_likelihood_x_tilde = self.get_likelihood(logits_score, samples)
        miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
        sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
        discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
        return discrepancy.item()
    def get_samples(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        nsamples = 10000
        lprobs = torch.log_softmax(logits, dim=-1)
        distrib = torch.distributions.categorical.Categorical(logits=lprobs)
        samples = distrib.sample([nsamples]).permute([1, 2, 0])
        return samples

    def get_likelihood(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        return log_likelihood.mean(dim=1)

    def __call__(self, data, sampled_data=None):
        if self.args.use_for_score:
            results = []
            n_samples = len(data)
            for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {self.name} criterion"):
                sampled_text = data[idx]
                # sampled text
                tokenized = self.scoring_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                                   return_token_type_ids=False).to(self.args.device)
                labels = tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits_score = self.scoring_model(**tokenized).logits[:, :-1]
                    if self.args.reference_model_name == self.args.scoring_model_name:
                        logits_ref = logits_score
                    else:
                        tokenized = self.reference_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                                             return_token_type_ids=False).to(self.args.device)
                        assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                        logits_ref = self.reference_model(**tokenized).logits[:, :-1]
                    sampled_crit = self.criterion_fn(logits_ref, logits_score, labels)
                # result
                results.append(sampled_crit)
            return results
        else:
            results = []
            n_samples=len(data['real'])
            for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {self.name} criterion"):
                real_text = data["real"][idx]
                sampled_text = data["sampled"][idx]
                # real text
                tokenized = self.scoring_tokenizer(real_text, return_tensors="pt", padding=True,
                                          return_token_type_ids=False).to(self.args.device)
                labels = tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits_score = self.scoring_model(**tokenized).logits[:, :-1]
                    if self.args.reference_model_name == self.args.scoring_model_name:
                        logits_ref = logits_score
                    else:
                        tokenized = self.reference_tokenizer(real_text, return_tensors="pt", padding=True,
                                                    return_token_type_ids=False).to(self.args.device)
                        assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                        logits_ref = self.reference_model(**tokenized).logits[:, :-1]
                    real_crit = self.criterion_fn(logits_ref, logits_score, labels)
                # sampled text
                tokenized = self.scoring_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                          return_token_type_ids=False).to(self.args.device)
                labels = tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits_score = self.scoring_model(**tokenized).logits[:, :-1]
                    if self.args.reference_model_name == self.args.scoring_model_name:
                        logits_ref = logits_score
                    else:
                        tokenized = self.reference_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                                    return_token_type_ids=False).to(self.args.device)
                        assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                        logits_ref = self.reference_model(**tokenized).logits[:, :-1]
                    sampled_crit = self.criterion_fn(logits_ref, logits_score, labels)

                # result
                results.append({"real": real_text,
                            "real_crit": real_crit,
                            "sampled": sampled_text,
                            "sampled_crit": sampled_crit})
            return results
