import os
import glob
import pandas as pd
import numpy as np
import sphn

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset

from huggingface_hub import hf_hub_download
import torch
import torch.functional as F

from moshi.models import loaders, LMGen

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup Mimi
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME, token=True)
mimi = loaders.get_mimi(mimi_weight, device=device)
mimi.set_num_codebooks(32)  # up to 32 for mimi
pcm_chunk_size = int(mimi.sample_rate / mimi.frame_rate)
mimi_sample_rate = mimi.sample_rate

# Setup Whisper
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

input_features = processor(
    sample["array"], return_tensors="pt", truncation=False, sampling_rate=sample["sampling_rate"]
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




def load_and_split_audio(input_folder, output_file=None):
    """
    Load all .rttm files in the input folder, split the audio files into chunks, and tokenize the chunks.
    
    Args:
        input_folder (str): Path to the folder containing .rttm files
    """
    # Get all .rttm files in the folder
    rttm_files = ["small.rttm"]  # glob.glob(os.path.join(input_folder, "*.rttm"))

    rttm_files = ["/home/thomwolf/Documents/Github/notebooklm-sample/small.rttm"]
    output_file = "/home/thomwolf/Documents/Github/notebooklm-sample/small.rttm"

    for rttm_file in rttm_files:
        full_rttm_file = os.path.join(input_folder, rttm_file)
        # Load the rttm file in a pandas dataframe (separator is a space).
        # Load the rttm file in a pandas dataframe
        # The first column is the type, the second is the file id, the third is the channel id, the fourth is the turn onset, the fifth is the turn duration, the sixth is the orthography field, the seventh is the speaker type, the eighth is the speaker name, the ninth is the confidence score, and the tenth is the signal lookahead time.
        # We have no header, so we need to specify the header names.
        df = pd.read_csv(full_rttm_file, sep=" ", header=None, names=["type", "file_id", "channel_id", "turn_onset", "turn_duration", "orthography_field", "speaker_type", "speaker_name", "confidence_score", "signal_lookahead_time"])

        # Load the wav files and split it based on the turn onsets and turn durations.
        # Turn Onset -- onset of turn in seconds from beginning of recording
        # Turn Duration -- duration of turn in seconds

        # Load the wav file with sphn and get it in a numpy array.
        # we change the extension of the file to .wav:
        wav_file = rttm_file.replace(".rttm", ".wav")
        sample_pcm, sample_pcm_source_rate = sphn.read(wav_file)
        print("loaded pcm", sample_pcm.shape[-1] / sample_pcm_source_rate)

        # Extract speaker turns from wav
        dataset = []
        for i in range(len(df)):
            speaker_id = df['speaker_name'][i]
            turn_onset = df['turn_onset'][i]
            turn_duration = df['turn_duration'][i]
            chunk_start = int(turn_onset * sample_pcm_source_rate)
            chunk_end = int((turn_onset + turn_duration) * sample_pcm_source_rate)

            # if (chunk_end - chunk_start}/sample_pcm_source_rate * mimi.frame_rate % 0 != 0:
            #  Make it a multiple of mimi.frame_rate

            turn_pcm = sample_pcm[:, chunk_start:chunk_end]

            # Resample and make mono
            turn_pcm = sphn.resample(
                turn_pcm, src_sample_rate=sample_pcm_source_rate, dst_sample_rate=mimi_sample_rate
            )
            # Make it mono by averaging the channels.
            turn_pcm = np.mean(turn_pcm, axis=0)
            turn_pcm = torch.tensor(turn_pcm, device=device)

            with torch.no_grad():
                codes = mimi.encode(turn_pcm)  # [B, K = 32, T]
                text = whisper.encode(turn_pcm)
            
            if speaker_id == "SPEAKER_00":
                speaker_int = 0
            elif speaker_id == "SPEAKER_01":
                speaker_int = 1
            else:
                raise ValueError() 
            dataset.append((speaker_int, text.squeeze(dim=0), codes.squeeze(dim=0)))

        # max_length = max(turn.shape[-1] for turn in speaker_turns)
        # speaker_turns = [F.pad(turn, (0, max_length - turn.shape[-1])) for turn in speaker_turns]
        # batch_turns = torch.stack(speaker_turns, dim=0).to(device=device)
        # with torch.no_grad():
        #     batch_codes = mimi.encode(batch_turns)  # [B, K = 32, T]


        ## WARNING: When streaming, make sure to always feed a total amount of audio that is a multiple
        #           of the frame size (1920), otherwise the last frame will not be complete, and thus
        #           will not be encoded. For simplicity, we recommend feeding in audio always in multiple
        #           of the frame size, so that you always know how many time steps you get back in `codes`.



if __name__ == "__main__":
    import argparse
    
    # parser = argparse.ArgumentParser(description="Concatenate all .rttm files in a folder")
    # parser.add_argument("input_folder", help="Path to the folder containing .rttm files")
    # parser.add_argument("output_file", help="Path to the output file")
    
    # args = parser.parse_args()
    
    # load_and_split_audio(args.input_folder, args.output_file)

    load_and_split_audio(input_folder="../notebooklm-sample", output_file="../notebooklm-sample/output")




# Example usage
# python split-an.py /path/to/your/folder output.rttm