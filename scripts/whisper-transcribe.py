import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset
from pydub import AudioSegment
import wave
from scipy.signal import resample
import librosa


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

wav_file = "/home/thomwolf/Documents/Github/notebooklm-sample/small.wav"
new_rate = 16000

y, sr = librosa.load(wav_file, sr=new_rate)

input_features = processor(
    y, return_tensors="pt", truncation=False, sampling_rate=new_rate
).input_features
input_features = input_features.to(device, torch_dtype)

generate_kwargs = {
    "language": "en",
    "max_new_tokens": 445,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    # "no_speech_threshold": 0.6,
    "return_timestamps": True,
    "return_segments": True,
}

generated_ids = model.generate(input_features, **generate_kwargs)
segments = generated_ids["segments"][0]

for segment in segments:
    text = processor.decode(segment["tokens"], skip_special_tokens=True)
    print(f"start: {segment['start']}, end: {segment['end']}, text: {text}")
