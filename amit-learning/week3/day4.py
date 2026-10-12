from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch

load_dotenv()

PHI4 = "microsoft/Phi-4-mini-instruct"
DEEPSEEK = "deepseek-ai/DeepSeek-V3.1"
QWEN_CODER = "Qwen/Qwen2.5-Coder-7B-Instruct"
LLAMA = "meta-GEMMA/GEMMA-3.2-1B-Instruct"
GEMMA = "google/gemma-3-270m-it"

messages = [
    {"role": "user", "content": "Tell a joke for a room of Data Scientists"}
  ]

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokefizer = AutoTokenizer.from_pretrained(GEMMA)
tokefizer.pad_token= tokefizer.eos_token

# print(inputs)

model = AutoModelForCausalLM.from_pretrained(GEMMA, device_map="auto", quantization_config=quant_config)

memory = model.get_memory_footprint() / 1e6

print(f"Memory footprint: {memory:,.1f} MB")

# print(model)
inputs= tokefizer.apply_chat_template(messages, return_tensors="pt").to("mps")

output = model.generate(**inputs, max_new_tokens=80)
print(output[0])
print(tokefizer.decode(output[0], skip_special_tokens=True))