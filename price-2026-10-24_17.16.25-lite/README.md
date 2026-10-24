---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: transformers
model_name: price-2026-10-24_17.16.25-lite
tags:
- generated_from_trainer
- sft
- trl
licence: license
---

# Model Card for price-2026-10-24_17.16.25-lite

This model is a fine-tuned version of [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="amitdeshmukh/price-2026-10-24_17.16.25-lite", device_map="auto")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://forge.coreweave.com/wandb/amiitdeshmukh-amit-s-udemy/price/runs/4ozdz5my) 



This model was trained with SFT.

### Framework versions

- TRL: 1.16.0
- Transformers: 5.19.0
- Pytorch: 2.14.1
- Datasets: 5.1.0
- Tokenizers: 0.23.3

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```