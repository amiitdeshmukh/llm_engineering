import gradio as gr
import locale
import modal
from agents.preprocessor import Preprocessor
from dotenv import load_dotenv

from llama import app, generate

with modal.enable_output():
    with app.run():
        result=generate.remote("Never gonna give you up, never gonna")

print(result)