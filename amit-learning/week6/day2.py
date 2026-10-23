from litellm import completion
from dotenv import load_dotenv
import json, os
from pricer.batch import Batch
from pricer.items import Item
from openai import OpenAI

load_dotenv()

LITE_MODE = True

username = "ed-donner"
dataset = f"{username}/items_raw_lite" if LITE_MODE else f"{username}/items_raw_full"

train, val, test = Item.from_hub(dataset)

items = train + val + test

# # print(f"Loaded {len(items):,} items")


# item = items[0]

# print(f"Title:    {item.title}")
# print(f"Category: {item.category}")
# print(f"Price:    ${item.price:.2f}")
# print(f"Weight:   {item.weight} pounds")

# print("\nProduct text:")
# print(item.full)

# print("\nOther fields:")
# print(f"Summary: {item.summary}")
# print(f"Prompt:  {item.prompt}")
# print(f"ID:      {item.id}")

for index, item in enumerate(items):
    item.id= index


SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""

# print(items[0].full)

# messages = [
#     {"role":"system", "content": SYSTEM_PROMPT},
#     {"role":"user", "content": items[0].full}
# ]
# response= completion(messages=messages, model='gpt-6-luna', reasoning_effort="low")

# print(response.choices[0].message.content)
# print()
# print(f"Input tokens: {response.usage.prompt_tokens}")
# print(f"Output tokens: {response.usage.completion_tokens}")
# print(f"Cost: {response._hidden_params['response_cost']*100:.3f} cents")

# MODEL="gpt-6-luna"

# def make_jsonl(item):
#     body = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item.full}], "reasoning_effort": "low"}
#     line = {"custom_id": str(item.id), "method": "POST", "url": "/v1/chat/completions", "body": body}
#     return json.dumps(line)

# def make_file(start, end, filename):
#     batch_file = filename
#     with open(batch_file, "w", encoding="utf-8") as f:
#         for i in range(start, end):
#             f.write(make_jsonl(items[i]))
#             f.write("\n")

# make_file(0,1000, "jsonl/0_1000.jsonl")

# openai = OpenAI()

# with open("jsonl/0_1000.jsonl", "rb") as f:
#     response = openai.files.create(file=f, purpose="batch")

# file_id=response.id

# response= openai.batches.create(completion_window="24h", endpoint="/v1/chat/completions", input_file_id=file_id)

# result = openai.batches.retrieve(response.id)
# response.write_to_file("jsonl/batch_results.jsonl")

# print("Batch ID:", response.id)
# print("Status:", response.status)


with open("jsonl/batch_results.jsonl", 'r', encoding="utf-8") as f:
    for line in f:
        json_line = json.loads(line)
        id = int(json_line["custom_id"])
        summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
        items[id].summary = summary

for item in items:
    print(item.title)
