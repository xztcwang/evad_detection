import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class Commercial_Detectors:
    def __init__(self, args):
        self.args=args
        if self.args.detection_method=="roberta-large":
            self.detector_mod = AutoModelForSequenceClassification.from_pretrained('roberta-large-openai-detector',
                                                                          cache_dir=args.cache_dir).to(self.args.device)
            self.detector_tok = AutoTokenizer.from_pretrained('roberta-large-openai-detector', cache_dir=args.cache_dir)
        if self.args.detection_method=="roberta-base":
            self.detector_mod = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector',
                                                                                   cache_dir=args.cache_dir).to(
                self.args.device)
            self.detector_tok = AutoTokenizer.from_pretrained('roberta-base-openai-detector', cache_dir=args.cache_dir)
        self.detector = pipeline('text-classification', model=self.detector_mod, tokenizer=self.detector_tok,
                            batch_size=args.detect_batch_size, device=self.args.device)

    def __call__(self, data, truncation=True):
        if self.args.use_for_score:
            results = []
            outs = self.detector(data, truncation=True)
            if self.args.detection_method=="roberta-base":
                results.extend(
                    [o['score'] if o['label'] == 'Real' else 1 - o['score'] for o in outs])  # 'Real' is human
            if self.args.detection_method == "roberta-large":
                results.extend([o['score'] if o['label'] == 'LABEL_1' else 1 - o['score'] for o in outs]) # 'LABEL_1' is human
            return results
        else:
            data_all = data["real"] + data["sampled"]
            inputs = self.detector_tok(data_all, return_tensors='pt', padding=True,
                                       truncation=True, max_length=self.args.m_length).to(self.args.device)
            with torch.no_grad():
                outputs = self.detector_mod(**inputs)
                logits = outputs.logits
            results = torch.softmax(logits.cpu(), dim=1)[:, 1].numpy()
            return results



