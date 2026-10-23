import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import random
from pricer.items import Item
from pricer.parser import parse


# load dataset-----------------

# dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)
# print(f"Number of Appliances: {len(dataset):,}")
# print(dataset[10])

# max_price =0
# max_item= None

# for datapoint in tqdm(dataset):
#     try:
#         price = float(datapoint["price"])
#         if price > max_price:
#             max_item=datapoint
#             max_price=price
#     except ValueError:
#         pass

# print(f"{max_item['title']} costs {max_price:,.2f}")

# items = [parse(datapoint, "Appliances") for datapoint in tqdm(dataset)]
# items =[item for item in items if item is not None]
# print(f"There are {len(items):,} items from {len(dataset):,} datapoints")

# print(items[0])

# prices = [item.price for item in items]
# lengths = [len(item.full) for item in items]

# plt.figure(figsize=(15,6))
# plt.title(f"Lengths: Avg {sum(lengths)/len(lengths):,.0f} and highest {max(lengths):,}\n")
# plt.xlabel('Length (chars)')
# plt.ylabel('Count')
# plt.hist(lengths, rwidth=0.5, color="lightblue", bins=range(0,6000,100))
# plt.show()

#  ----------------------------------

# max_length = max(lengths)
# max_length_item = items[lengths.index(max_length)]
# print(max_length_item.full)


# plt.figure(figsize=(15, 6))
# plt.title(f"Prices: Avg {sum(prices)/len(prices):,.2f} and highest {max(prices):,}\n")
# plt.xlabel('Price ($)')
# plt.ylabel('Count')
# plt.hist(prices, rwidth=0.7, color="orange", bins=range(0, 1000, 10))
# plt.show()

from pricer.loaders import ItemLoader

dataset_names = [
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Appliances",
    "Musical_Instruments",
]


def main():

    load_dotenv()

    hf_token = os.environ['HF_TOKEN']
    login(hf_token, add_to_git_credential=True)

    items = []
    for dataset_name in dataset_names:
        loader = ItemLoader(dataset_name)
        items.extend(loader.load())

    print(f"A grand total of {len(items):,} items")

if __name__ == "__main__":
    main()