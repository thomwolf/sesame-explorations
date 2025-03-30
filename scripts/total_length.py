import wave
import os
import glob

def calculate_wav_files_length(directory_path):
    """
    Calculate the total length (in seconds) of all WAV files in a directory.
    
    Parameters:
    directory_path (str): Path to directory containing WAV files
    
    Returns:
    float: Total length of all WAV files in seconds
    """
    total_duration = 0
    
    # Get all WAV files from directory using glob (non-recursive)
    wav_files = glob.glob(os.path.join(directory_path, "*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {directory_path}")
        return 0
    
    print(f"Found {len(wav_files)} WAV files in {directory_path}")
    
    # Process each WAV file
    for wav_file in wav_files:
        try:
            with wave.open(wav_file, 'r') as wf:
                # Calculate duration: frames / framerate
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                total_duration += duration
                print(f"{os.path.basename(wav_file)}: {duration:.2f} seconds")
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
    
    # Convert to hours, minutes, seconds format
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTotal duration: {int(hours)}:{int(minutes):02d}:{seconds:.2f}")
    return total_duration

# Example usage
if __name__ == "__main__":
    # Replace with your actual directory path
    directory_path = "./data"
    calculate_wav_files_length(directory_path)
