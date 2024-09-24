import torch
import torch.nn.functional as F
import tqdm
from utils.model_utils import load_model,load_tokenizer
class Baseline_Detectors:
    def __init__(self, args):
        self.args=args
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir, use_ft=False)
        self.scoring_model = load_model(args, args.scoring_model_name, args.device,
                                   args.cache_dir, use_ft=False, data_generation_flag=True)

        self.scoring_model.resize_token_embeddings(len(self.scoring_tokenizer))

        self.scoring_model.eval()
        if self.args.detection_method=="Likelihood":
            self.baseline_fn=self.get_likelihood
        elif self.args.detection_method=="Entropy":
            self.baseline_fn =self.get_entropy
        elif self.args.detection_method == "LogRank":
            self.baseline_fn =self.get_logrank
        elif self.args.detection_method == "Rank":
            self.baseline_fn =self.get_rank
        else:
            self.baseline_fn =self.get_likelihood


    def get_likelihood(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1

        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return log_likelihood.mean().item()

    def get_rank(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        return -ranks.mean().item()

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
        return -ranks.mean().item()

    def get_entropy(self,logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1

        entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        entropy = -entropy.sum(-1)
        return entropy.mean().item()
    def __call__(self, data, sampled_data=None):
        if self.args.use_for_score:
            results = []
            n_samples = len(data)
            for idx in tqdm.tqdm(range(n_samples), desc=f"computing baseline detector"):
                sampled_text = data[idx]
                # sampled text
                sampled_tokenized = self.scoring_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                                           return_token_type_ids=False).to(self.args.device)
                sampled_labels = sampled_tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits = self.scoring_model(**sampled_tokenized).logits[:, :-1]
                    sampled_crit = self.baseline_fn(logits, sampled_labels)
                # result
                results.append(sampled_crit)
            return results
        else:
            results = []
            n_samples = len(data['real'])
            for idx in tqdm.tqdm(range(n_samples), desc=f"computing baseline detector"):
                real_text = data["real"][idx]
                sampled_text = data["sampled"][idx]
                # real text
                real_tokenized = self.scoring_tokenizer(real_text, return_tensors="pt", padding=True,
                                                        return_token_type_ids=False).to(self.args.device)
                real_labels = real_tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits = self.scoring_model(**real_tokenized).logits[:, :-1]
                    real_crit = self.baseline_fn(logits, real_labels)
                # sampled text
                sampled_tokenized = self.scoring_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                                               return_token_type_ids=False).to(self.args.device)
                sampled_labels = sampled_tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits = self.scoring_model(**sampled_tokenized).logits[:, :-1]
                    sampled_crit = self.baseline_fn(logits, sampled_labels)
                # result
                results.append({"real": real_text,
                                    "real_crit": real_crit,
                                    "sampled": sampled_text,
                                    "sampled_crit": sampled_crit})
            return results




