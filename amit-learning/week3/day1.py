import torch
import soundfile as sf
from transformers import pipeline
from datasets import load_dataset
from dotenv import load_dotenv
import os

load_dotenv()

device = "mps"
print("HF token loaded:", bool(os.getenv("HF_TOKEN")))

# 2. Initialize the TTS pipeline
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device)

# 3. Load the speaker embeddings dataset
embeddings_dataset = load_dataset("matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# 4. Generate the speech
speech = synthesiser(
    "Hi to an artificial intelligence engineer, on the way to mastery!",
    forward_params={"speaker_embeddings": speaker_embedding}
)

# 5. Save the generated audio to a WAV file
output_filename = "output.wav"
sf.write(output_filename, speech["audio"], speech["sampling_rate"])
print(f"Audio successfully saved to {output_filename}")
