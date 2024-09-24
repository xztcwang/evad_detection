## Code for ICLR 2025 Submission

### Requirements

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

