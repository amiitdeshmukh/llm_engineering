import os
import re
import math
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
from datetime import datetime

load_dotenv()

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
PROJECT_NAME = "price"
RUN_NAME =  f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"

LITE_MODE = False

DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"


FINETUNED_MODEL = f"ed-donner/price-2025-11-30_15.10.55-lite"



hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)

# base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="mps")
# print(f"Memory footprint: {base_model.get_memory_footprint()/1e9:,.1f} GB")

# print(base_model)


quant_config = BitsAndBytesConfig(load_in_8bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="mps"
)

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")

print(base_model)

r=32

# See the matrix dimensions above

# Attention layers
lora_q_proj = 2048 * r + 2048 * r
lora_k_proj = 2048 * r + 256 * r
lora_v_proj = 2048 * r + 256 * r
lora_o_proj = 2048 * r + 2048 * r

# Each layer comes to
lora_layer = lora_k_proj + lora_o_proj + lora_q_proj + lora_v_proj


params = lora_layer*36
size = (params * 4) / 1_000_000

print(f"Total number of params: {params:,} and size {size:,.1f}MB")