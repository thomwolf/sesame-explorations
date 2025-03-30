import os
import glob
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=True)

# send pipeline to GPU (when available)
# import torch
# pipeline.to(torch.device("cuda"))

# list .wav files in the directory
wav_files = ["./data/small.wav"] + glob.glob("./data/*.wav")  # ["./data/small.wav"]  # 
for wav_file in wav_files:
    # check if the file exists
    if os.path.exists(wav_file.replace('.wav', '.rttm')):
        print(f"File {wav_file} already processed, skipping...")
        continue
    # check if the file exists
    if not os.path.exists(wav_file):
        print(f"File {wav_file} does not exist, skipping...")
        break
    # apply pretrained pipeline
    print(f"Starting diarization on file {wav_file}...")
    diarization = pipeline(wav_file, num_speakers=2)
    print("Finished diarization...")
    # print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    # save the result to a file
    print("Saving diarization result...")
    with open(f"./data/{wav_file.split('/')[-1].replace('.wav', '')}.rttm", "w") as f:
        diarization.write_rttm(f)
    print(f"Finished saving diarization of {wav_file}...")

