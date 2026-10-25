import os
from dotenv import load_dotenv
from openai import OpenAI
from agents.deals import ScrapedDeal, DealSelection
import logging
import requests
load_dotenv(override=True)
openai = OpenAI()
MODEL = 'gpt-6-luna'

deals = ScrapedDeal.fetch(show_progress=True)

# print(len(deals))

# print(deals[10].describe())

SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
"""

USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
You should rephrase the description to be a summary of the product itself, not the terms of the deal.
Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 

Deals:

"""

USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

def make_user_prompt(scraped):
    user_prompt = USER_PROMPT_PREFIX
    user_prompt += '\n\n'.join([scrape.describe() for scrape in scraped])
    user_prompt += USER_PROMPT_SUFFIX
    return user_prompt

user_prompt = make_user_prompt(deals)
print(user_prompt[:2000])
messages =[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_prompt}
]

response = openai.chat.completions.parse(model=MODEL, messages=messages, response_format=DealSelection, reasoning_effort='low')

results = response.choices[0].message.parsed
print(results.model_dump_json(indent=2))