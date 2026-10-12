import torch
from dotenv import load_dotenv
import os
from huggingface_hub import login
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
import soundfile as sf

load_dotenv()

# sentiment_analyzer = pipeline("sentiment-analysis", device="mps")
# result = sentiment_analyzer("I'm not excited to be on the way to LLM mastery!")
# print(result)

# ner = pipeline("ner", device="mps")
# result = ner("AI Engineers are learning about the amazing pipelines from HuggingFace in Google Colab from Ed Donner")
# for entity in result:
#   print(entity)

question="What are Hugging Face pipelines?"
context="Pipelines are a high level API for inference of LLMs with common tasks"

question_answerer = pipeline("question-answering", device="cuda")
result = question_answerer(question=question, context=context)
print(result)