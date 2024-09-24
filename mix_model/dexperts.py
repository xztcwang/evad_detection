import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from typing import List

class DExpertsLlama:
    def __init__(
        self,
        large_model,
        small_model,
        small_finetune_model,
        tokenizer: AutoTokenizer,
        alpha: float = 1.0,
        max_seq_len:int=384,
        mix_ratio: float=1.0
    ):
        self.large_model = large_model
        self.small_model = small_model
        self.small_finetune_model = small_finetune_model
        self.large_model.eval()
        self.small_model.eval()
        self.small_finetune_model.eval()

        self.tokenizer = tokenizer
        self.mix_ratio=mix_ratio
        self.device = self.large_model.device
        self.max_seq_len = max_seq_len
        self.pad_token_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def generate_with_ref(
            self,
            encoded_tokens,
            cache_path: str,
            max_gen_len: int,
            temperature: float = 1.0,
            top_p: float = 1.0,
            beta: float = 1.0) -> List[str]:
        bsz = len(encoded_tokens)
        min_prompt_size = min([t['input_ids'].shape[1] for t in encoded_tokens])
        max_prompt_size = max([t['input_ids'].shape[1] for t in encoded_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((bsz, total_len), self.pad_token_id).to(self.device).long()
        for k, t in enumerate(encoded_tokens):
            tokens[k, : t['input_ids'].shape[1]] = torch.tensor(t['input_ids']).long()
        input_mask = tokens != self.pad_token_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            large_model_outputs = self.large_model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True,
                past_key_values=large_model_outputs.past_key_values if prev_pos > 0 else None
                )
            small_model_outputs = self.small_model.forward(
                tokens[:, prev_pos:cur_pos].to(self.small_model.device).long(), use_cache=True,
                past_key_values=small_model_outputs.past_key_values if prev_pos > 0 else None
            )
            small_finetune_model_outputs = self.small_finetune_model.forward(
                tokens[:, prev_pos:cur_pos].to(self.small_finetune_model.device).long(), use_cache=True,
                past_key_values=small_finetune_model_outputs.past_key_values if prev_pos > 0 else None
            )
            if temperature > 0:
                ori_lprobs = torch.log_softmax(large_model_outputs.logits[:, -1, :] / temperature, dim=-1)
                small_model_lprobs = torch.log_softmax(small_model_outputs.logits[:, -1, :] / temperature, dim=-1).to(
                self.device)
                if small_finetune_model_outputs.logits.shape[2] == 32001:
                    small_finetune_model_lprobs = torch.log_softmax(
                        small_finetune_model_outputs.logits[:, -1, :-1] / temperature, dim=-1).to(self.device)
                else:
                    small_finetune_model_lprobs = torch.log_softmax(
                        small_finetune_model_outputs.logits[:, -1, :] / temperature, dim=-1).to(self.device)
            else:
                ori_lprobs = torch.log_softmax(large_model_outputs.logits[:, -1, :], dim=-1)
                small_model_lprobs = torch.log_softmax(small_model_outputs.logits[:, -1, :], dim=-1).to(self.device)
                small_finetune_model_lprobs = torch.log_softmax(small_finetune_model_outputs.logits[:, -1, :], dim=-1).to(
                    self.device)
            new_lprobs = ori_lprobs + self.mix_ratio * (small_finetune_model_lprobs - small_model_lprobs)
            # Get normalizing constant
            log_normalizer = torch.logsumexp(new_lprobs, dim=-1, keepdim=True)
            # Subtract normalizing constant
            new_lprobs -= log_normalizer
            estimated_probs = torch.exp(new_lprobs)
            next_toks = self.sample_next_with_ref(estimated_probs, temperature, top_p)
            tokens[:, cur_pos] = torch.where(input_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            prev_pos = cur_pos
        cut_tokens = []
        for i, t in enumerate(tokens.tolist()):
            t = t[: len(tokens[i]) + self.max_seq_len]
            cut_tokens.append(torch.tensor(t))


        return cut_tokens

    def sample_next_with_ref(
            self,
            probs: torch.FloatTensor,  # (bsz, vocab_size): logits for last token
            temperature: float = 1.0,  # temperature for sampling
            top_p: float = 1.0,  # top p for sampling
    ) -> torch.LongTensor:
        if temperature > 0:
            try:
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                mask = probs_sum - probs_sort > top_p
                probs_sort[mask] = 0.0
                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                next_token = torch.multinomial(probs_sort, num_samples=1)  # one hot of next token, ordered by original probs
                next_token = torch.gather(probs_idx, -1, next_token)  # one hot of next token, ordered by vocab
            except:
                next_token = torch.argmax(probs, dim=-1)
        else:
            next_token = torch.argmax(probs, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

