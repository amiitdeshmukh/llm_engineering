import os, requests, torch
from pathlib import Path
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from dotenv import load_dotenv
import mlx_whisper


GEMMA = "google/gemma-3-270m-it"
WHISPER="mlx-community/whisper-large-v3-turbo"

PROJECT_ROOT= Path(__file__).resolve().parents[2]
AUDIO_FILE=Path(__file__).with_name("denver_extract.mp3")
OUTPUT_FILE=Path(__file__).with_name("mom.md")
TRANSCRIPTION_FILE=Path(__file__).with_name("transcription.txt")

if not AUDIO_FILE.exists():
    raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")

load_dotenv(PROJECT_ROOT / ".env")
hf_token=os.getenv("HF_TOKEN")

if hf_token:
    login(
        token=hf_token,
        add_to_git_credential=False
    )

print("Loading Whisper and transcribing the audio...")
print("This can take several minutes.\n")

transcription_result= mlx_whisper.transcribe(
    str(AUDIO_FILE),
    path_or_hf_repo=WHISPER,
)

transcription = transcription_result["text"].strip()

if not transcription:
    raise RuntimeError("Whisper returned empty results")

TRANSCRIPTION_FILE.write_text(
    transcription,
    encoding="utf-8"
)

device = "mps"

tokenizer = AutoTokenizer.from_pretrained(
    GEMMA,
    token=hf_token
)

model = AutoModelForCausalLM.from_pretrained(
    GEMMA,
    token=hf_token,
    dtype=torch.float32
).to(device)

model.eval()

prompt = f"""
You create accurate meeting minutes from transcripts.

Read the transcript below and return Markdown with exactly these sections:

# Meeting Minutes

## Summary

## Attendees, Location, and Date

## Key Discussion Points

## Takeaways

## Action Items

For every action item, identify the owner when one is mentioned.
If the transcript does not provide some information, write "Not specified."
Do not invent names, dates, locations, decisions, or action items.

Transcript:

{transcription}
"""

messages=[
    {
        "role":"user","content":prompt
    }
]

inputs= tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to(device)

print("generating mom:")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

prompt_length = inputs["input_ids"].shape[1]
generated_tokens = output[0, prompt_length:]

minutes = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
).strip()

if not minutes:
    raise RuntimeError("Gemma returned empty meeting minutes.")



print("--- GENERATED MEETING MINUTES ---\n")
print(minutes)

OUTPUT_FILE.write_text(
    minutes,
    encoding="utf-8",
)

print(f"\nMeeting minutes saved to: {OUTPUT_FILE}")