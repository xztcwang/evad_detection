from utils.model_utils import load_model,load_tokenizer
import torch
import tqdm
SEPARATOR = '<<<SEP>>>'

class DNAGPT:
    def __init__(self, args):
        self.args=args
        self.base_tokenizer = load_tokenizer(args.base_model_name, args.dataset, args.cache_dir, use_ft=False)
        self.base_model = load_model(args, args.base_model_name, args.device,
                                         args.cache_dir, use_ft=False, data_generation_flag=True)
        self.base_model.eval()

        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir, use_ft=False)
        if self.args.scoring_model_name==self.args.base_model_name:
            self.scoring_model = load_model(args, args.scoring_model_name, args.device,
                                        args.cache_dir, use_ft=False, data_generation_flag=True)
        else:
            self.scoring_model = load_model(args, args.scoring_model_name, args.device,
                                            args.cache_dir, use_ft=False, data_generation_flag=False)
        self.scoring_model.eval()

    def get_likelihood(self, logits, labels, pad_index):
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        mask = labels != pad_index
        log_likelihood = (log_likelihood * mask).sum(dim=1) / mask.sum(dim=1)
        return log_likelihood.squeeze(-1)


    def get_log_prob(self, text):
        tokenized = self.scoring_tokenizer(text, return_tensors="pt", padding=True).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            return self.get_likelihood(logits_score, labels, self.scoring_tokenizer.pad_token_id)

    def get_log_probs(self, texts):
        batch_size = self.args.batch_size
        batch_lprobs = []
        for batch in range(len(texts) // batch_size):
            tokenized = self.scoring_tokenizer(texts[batch * batch_size:(batch + 1) * batch_size], return_tensors="pt",
                                               padding=True).to(self.args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = self.scoring_model(**tokenized).logits[:, :-1]
                lprobs = self.get_likelihood(logits_score, labels, self.scoring_tokenizer.pad_token_id)
                batch_lprobs.append(lprobs)
        return torch.cat(batch_lprobs, dim=0)

    def _sample_from_model(self, input_batch, min_words=55, truncate_ratio=0.5):
        if self.args.dataset == 'pubmed':
            pubmed_sep = ' Answer:'
            input_batch = [t[:t.index(pubmed_sep) + len(pubmed_sep)] for t in input_batch]
            all_encoded = self.base_tokenizer(input_batch, return_tensors="pt", padding=True).to(self.args.device)
        else:
            input_batch = [t.split(' ') for t in input_batch]
            input_batch = [' '.join(t[: int(len(t) * truncate_ratio)]) for t in input_batch]
            all_encoded = self.base_tokenizer(input_batch, return_tensors="pt", padding=True).to(self.args.device)

        self.base_model.eval()
        decoded = ['' for _ in range(len(input_batch))]
        tries = 0
        m = 0
        while m < min_words:
            if tries != 0:
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
            min_length = 50 if self.args.dataset in ['pubmed'] else 150
            sampling_kwargs = {'temperature': self.args.temperature}
            if self.args.do_top_p:
                sampling_kwargs['top_p'] = self.args.top_p
            elif self.args.do_top_k:
                sampling_kwargs['top_k'] = self.args.top_k
            outputs = self.base_model.generate(
                **all_encoded,
                min_length=min_length,
                max_length=200,
                do_sample=True,
                **sampling_kwargs,
                pad_token_id=self.base_tokenizer.eos_token_id,
                eos_token_id=self.base_tokenizer.eos_token_id)
            decoded = self.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            m = min(len(x.split()) for x in decoded)
            tries += 1
        return decoded

    def generate_samples(self,  input_data, batch_size=1):
        def _truncate_to_substring(text, substring, idx_occurrence):
            # truncate everything after the idx_occurrence occurrence of substring
            assert idx_occurrence > 0, 'idx_occurrence must be > 0'
            idx = -1
            for _ in range(idx_occurrence):
                idx = text.find(substring, idx + 1)
                if idx == -1:
                    return text
            return text[:idx]
        def _trim_to_shorter_length(texta, textb):
            # truncate to shorter of o and s
            shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
            texta = ' '.join(texta.split(' ')[:shorter_length])
            textb = ' '.join(textb.split(' ')[:shorter_length])
            return texta, textb

        sampled_data=[]
        for batch in range(len(input_data) // batch_size):
            print('Generating samples for batch', batch, 'of', len(input_data) // batch_size)
            input_batch = input_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_batch = self._sample_from_model(input_batch,
                                                   min_words=30 if self.args.dataset in ['pubmed'] else 55,
                                                   truncate_ratio=self.args.truncate_ratio)
            for o, s in zip(input_batch, sampled_batch):
                if self.args.dataset == 'pubmed':
                    s = _truncate_to_substring(s, 'Question:', 2)
                    o = o.replace(SEPARATOR, ' ')
                o, s = _trim_to_shorter_length(o, s)

                sampled_data.append(s)
        return sampled_data







    def get_regen_samples(self, text):
        data = [text] * self.args.regen_number
        s_data = self.generate_samples(data, batch_size=self.args.batch_size)
        return s_data

    def get_dna_gpt(self, text):
        lprob = self.get_log_prob(text)
        regens = self.get_regen_samples(text)
        lprob_regens = self.get_log_probs(regens)
        wscore = lprob[0] - lprob_regens.mean()
        return wscore.item()


    def __call__(self, data, sampled_data=None):
        if self.args.use_for_score:
            results = []
            n_samples = len(data)
            for idx in tqdm.tqdm(range(n_samples), desc=f"Computing dna-gpt criterion"):
                sampled_text = data[idx]
                # sampled text
                sampled_crit = self.get_dna_gpt(sampled_text)
                results.append(sampled_crit)
            return results
        else:
            results = []
            n_samples=len(data['real'])
            for idx in tqdm.tqdm(range(n_samples), desc=f"Computing dna-gpt criterion"):
                real_text = data["real"][idx]
                sampled_text = data["sampled"][idx]
                # real text
                real_crit = self.get_dna_gpt(real_text)
                # sampled text
                sampled_crit = self.get_dna_gpt(sampled_text)
                # result
                results.append({"real": real_text,
                                "real_crit": real_crit,
                                "sampled": sampled_text,
                                "sampled_crit": sampled_crit})
            return results