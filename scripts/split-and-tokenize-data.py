import os
import glob
import pandas as pd
import numpy as np
import sphn

from huggingface_hub import hf_hub_download
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from moshi.models import loaders, LMGen

mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device='cuda' if torch.cuda.is_available() else 'cpu')
mimi.set_num_codebooks(32)  # up to 32 for mimi

pcm_chunk_size = int(mimi.sample_rate / mimi.frame_rate)
mimi_sample_rate = mimi.sample_rate

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
whisper_sample_rate = 16000
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


def tokenize_audio(input_folder, output_file):
    """
    Load all .rttm files in the input folder, split the audio files into chunks, and tokenize the chunks.
    
    Args:
        input_folder (str): Path to the folder containing .rttm files
    """
    # Get all .rttm files in the folder
    rttm_files = glob.glob(os.path.join(input_folder, "*.rttm"))

    rttm_files = ["/home/thomwolf/Documents/Github/notebooklm-sample/small.rttm"]
    output_file = "/home/thomwolf/Documents/Github/notebooklm-sample/small.rttm"

    for rttm_file in rttm_files:
        # Load the rttm file in a pandas dataframe (separator is a space).
        # Load the rttm file in a pandas dataframe
        # The first column is the type, the second is the file id, the third is the channel id, the fourth is the turn onset, the fifth is the turn duration, the sixth is the orthography field, the seventh is the speaker type, the eighth is the speaker name, the ninth is the confidence score, and the tenth is the signal lookahead time.
        # We have no header, so we need to specify the header names.
        df = pd.read_csv(rttm_file, sep="\t", header=None, names=["type", "file_id", "channel_id", "turn_onset", "turn_duration", "orthography_field", "speaker_type", "speaker_name", "confidence_score", "signal_lookahead_time"])

        # Load the wav files and split it based on the turn onsets and turn durations.
        # Turn Onset -- onset of turn in seconds from beginning of recording
        # Turn Duration -- duration of turn in seconds

        # Load the wav file with sphn and get it in a numpy array.
        # we change the extension of the file to .wav:
        wav_file = rttm_file.replace(".rttm", ".wav")
        sample_pcm, sample_pcm_source_rate = sphn.read(wav_file)
        print("loaded pcm", sample_pcm.shape, sample_pcm_source_rate)
        mimi_sample_pcm = sphn.resample(
            sample_pcm, src_sample_rate=sample_pcm_source_rate, dst_sample_rate=mimi_sample_rate
        )
        whisper_sample_pcm = sphn.resample(
            sample_pcm, src_sample_rate=sample_pcm_source_rate, dst_sample_rate=whisper_sample_rate
        )
        mimi_sample_pcm = torch.tensor(mimi_sample_pcm, device=device)
        whisper_sample_pcm = torch.tensor(whisper_sample_pcm, device=device)
        # add batch dimension and change from 2 channels to 1 channel by averaging the channels
        mimi_sample_pcm = mimi_sample_pcm[None].to(device=device)
        mimi_sample_pcm = mimi_sample_pcm.mean(dim=1, keepdim=True)

        # max_duration_len = int(mimi_sample_rate * max_duration_sec)
        # if sample_pcm.shape[-1] > max_duration_len:
        #     sample_pcm = sample_pcm[..., :max_duration_len]
        print("resampled pcm", mimi_sample_pcm.shape, mimi_sample_rate)
        # sample_pcm = sample_pcm[None].to(device=device)

        with torch.no_grad():
            codes = mimi.encode(mimi_sample_pcm)  # [B, K = 32, T]
        
        print("codes", codes.shape)
        # splits the code in chunks for each speaker based on the turn onsets and turn durations as well as the pcm_chunk_size
        for i in range(len(df)):
            turn_onset = df['turn_onset'][i]
            turn_duration = df['turn_duration'][i]
            chunk_start = int(turn_onset * mimi_sample_rate)
            chunk_end = int((turn_onset + turn_duration) * mimi_sample_rate)
            chunk = codes[:, :, chunk_start:chunk_end]
            print("chunk", chunk.shape)

            # split the whisper chunks
            chunk_start = int(turn_onset * whisper_sample_rate)
            chunk_end = int((turn_onset + turn_duration) * whisper_sample_rate)
            chunk_whisper = whisper_sample_pcm[:, chunk_start:chunk_end]
            print("chunk whisper", chunk_whisper.shape)
            # encode the chunk with the whisper model
            input_features = processor(
                chunk_whisper.cpu().numpy(), return_tensors="pt", truncation=False, sampling_rate=whisper_sample_rate
            ).input_features
            input_features = input_features.to(device=device, dtype=torch_dtype)
            # generate the tokens
            generated_ids = model.generate(input_features, **generate_kwargs)
            # decode the tokens
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
            # Gather text from the chunks
            text = []
            for segment in generated_ids["segments"][0]:
                text.append(processor.decode(segment["tokens"], skip_special_tokens=True))
            # print the text
            print(f"speaker {df['speaker_name'][i]}: {text}")
            # save the text to a file
            with open(os.path.join(output_file, f"{df['file_id'][0]}_{i}.txt"), "w") as f:
                for t in text:
                    f.write(t + "\n")
            # save the codes to a file
            np.save(os.path.join(output_file, f"{df['file_id'][0]}_{i}.npy"), chunk.cpu().numpy())


            # test decoding the chunk and write it to a wav file
            decoded_pcm = mimi.decode(chunk)
            sphn.write_wav(os.path.join(output_file, f"{df['file_id'][0]}_{i}.wav"), decoded_pcm.cpu().numpy(), mimi_sample_rate)
            print(f"wrote {df['file_id'][0]}_{i}.wav with shape {decoded_pcm.shape} and length {decoded_pcm.shape[-1] / mimi_sample_rate} seconds for speaker {df['speaker_name'][i]}")


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
    
    tokenize_audio("", "")

# Example usage
# python split-an.py /path/to/your/folder output.rttm