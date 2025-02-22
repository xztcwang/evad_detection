# HUMPA
This repository contains the code for [Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors](https://arxiv.org/pdf/2410.19230) publised in ICLR 2025.
### Dependencies

- python 3.10
- pytorch 2.3.1+cu121
- transformers 4.41.0
- peft 0.12.0
- datasets 2.20.0
- scikit-learn 1.5.1
- flash_attn 2.6.3
- accelerate 0.33.0

### How to Use

#### 1. Data Preparation:
```bash
python get_data/processing/get_prompts.py
python get_data/processing/get_samples.py
python dpo_builders/baseline_builders/dpo_builder.py
```
#### 2. Fine-tune the Small Model:
```bash
python dpo_fine_tune.py
```
#### 3. Generate the humanoid texts from attacked model:
```bash
python detection_eval/eval_data_builders/eval_builder_benchmarks.py
```
#### 4. Evaluate the detectors like:
```bash
python detection_eval/baseline_eval.py
python detection_eval/fast_detectgpt_eval.py
```

## Citation

```bash
@article{wang2024humanizing,
  title={Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors},
  author={Wang, Tianchun and Chen, Yuanzhou and Liu, Zichuan and Chen, Zhanwen and Chen, Haifeng and Zhang, Xiang and Cheng, Wei},
  journal={arXiv preprint arXiv:2410.19230},
  year={2024}
}
```
