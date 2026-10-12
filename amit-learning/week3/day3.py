from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

load_dotenv()

PHI4 = "microsoft/Phi-4-mini-instruct"
DEEPSEEK = "deepseek-ai/DeepSeek-V3.1"
QWEN_CODER = "Qwen/Qwen2.5-Coder-7B-Instruct"
model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK, trust_remote_code=True)
# text = "I am excited to show Tokenizers in action to my LLM engineers"
# tokens = tokenizer.encode(text)
# print(tokens)

# character_count = len(text)
# word_count = len(text.split(' '))
# token_count = len(tokens)
# print(f"There are {character_count} characters, {word_count} words and {token_count} tokens")

# print(tokenizer.decode(tokens))

# print(tokenizer.batch_decode(tokens))

# print(tokenizer.get_added_vocab())

# print(len(tokenizer.vocab))

#---------------------------------------------------------------


messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]

# prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print(prompt)

#---------------------------------------------------------------

phi4_tokenizer = AutoTokenizer.from_pretrained(PHI4)
text = "I am curiously excited to show Hugging Face Tokenizers in action to my LLM engineers"
print("distilbert:")
tokens = tokenizer.encode(text)
print(tokens)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
# for token_id, piece in zip(tokens, tokenizer.convert_ids_to_tokens(tokens)):
#     print(token_id, repr(piece))


print("\nPhi 4:")
tokens = phi4_tokenizer.encode(text)
print(tokens)
prompt = phi4_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
# for token_id, piece in zip(tokens, tokenizer.convert_ids_to_tokens(tokens)):
#     print(token_id, repr(piece))