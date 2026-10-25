import gradio as gr
import locale
import modal
from agents.preprocessor import Preprocessor
from dotenv import load_dotenv
from agents.specialist_agent import SpecialistAgent



# pricer = modal.Function.from_name("pricer-service", "price")

# preprocessor = Preprocessor()
# text = preprocessor.preprocess("Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio")
# print(pricer.remote(text))


agent = SpecialistAgent()
print(agent.price("iPhone 10"))