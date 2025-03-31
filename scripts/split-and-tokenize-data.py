import os
import glob
import pandas as pd
import numpy as np
import sphn

from huggingface_hub import hf_hub_download
import torch

from moshi.models import loaders, LMGen

mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device='cuda' if torch.cuda.is_available() else 'cpu')
mimi.set_num_codebooks(32)  # up to 32 for mimi

pcm_chunk_size = int(mimi.mimi_sample_rate / mimi.frame_rate)
mimi_sample_rate = mimi.mimi_sample_rate

def load_and_split_audio(input_folder, output_file):
    """
    Load all .rttm files in the input folder, split the audio files into chunks, and tokenize the chunks.
    
    Args:
        input_folder (str): Path to the folder containing .rttm files
    """
    # Get all .rttm files in the folder
    rttm_files = glob.glob(os.path.join(input_folder, "*.rttm"))

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
        wav_file = os.path.join(input_folder, f"{df['file_id'][0]}.wav")
        sample_pcm, sample_pcm_source_rate = sphn.read(wav_file)
        print("loaded pcm", sample_pcm.shape, sample_pcm_source_rate)
        sample_pcm = sphn.resample(
            sample_pcm, src_sample_rate=sample_pcm_source_rate, dst_sample_rate=mimi_sample_rate
        )
        sample_pcm = torch.tensor(sample_pcm, device=args.device)
        # max_duration_len = int(mimi_sample_rate * max_duration_sec)
        # if sample_pcm.shape[-1] > max_duration_len:
        #     sample_pcm = sample_pcm[..., :max_duration_len]
        print("resampled pcm", sample_pcm.shape, sample_pcm_source_rate)
        sample_pcm = sample_pcm[None].to(device=args.device)

        with torch.no_grad():
            codes = mimi.encode(sample_pcm)  # [B, K = 32, T]
        
        print("codes", codes.shape)
        # splits the code in chunks for each speaker based on the turn onsets and turn durations as well as the pcm_chunk_size
        for i in range(len(df)):
            turn_onset = df['turn_onset'][i]
            turn_duration = df['turn_duration'][i]
            chunk_start = int(turn_onset * mimi_sample_rate)
            chunk_end = int((turn_onset + turn_duration) * mimi_sample_rate)
            chunk = codes[:, :, chunk_start:chunk_end]
            print("chunk", chunk.shape)

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
    
    parser = argparse.ArgumentParser(description="Concatenate all .rttm files in a folder")
    parser.add_argument("input_folder", help="Path to the folder containing .rttm files")
    parser.add_argument("output_file", help="Path to the output file")
    
    args = parser.parse_args()
    
    concatenate_rttm_files(args.input_folder, args.output_file)

# Example usage
# python split-an.py /path/to/your/folder output.rttm