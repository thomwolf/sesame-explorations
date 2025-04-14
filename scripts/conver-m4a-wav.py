from pydub import AudioSegment
import os
import glob

def convert_m4a_to_wav(input_directory, output_directory=None):
    """
    Convert all M4A files in a directory to WAV format.
    
    Parameters:
    input_directory (str): Path to directory containing M4A files
    output_directory (str, optional): Path to output directory for WAV files. 
                                     If None, WAV files will be created in the same directory.
    """
    # Create output directory if it doesn't exist
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Get all M4A files from directory
    m4a_files = glob.glob(os.path.join(input_directory, "*.m4a"))
    
    if not m4a_files:
        print(f"No M4A files found in {input_directory}")
        return
    
    print(f"Found {len(m4a_files)} M4A files in {input_directory}")
    
    # Process each M4A file
    for m4a_file in m4a_files:
        try:
            # Get the base filename without extension
            base_name = os.path.basename(m4a_file)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # Determine output file path
            if output_directory:
                wav_file = os.path.join(output_directory, f"{name_without_ext}.wav")
            else:
                wav_file = os.path.join(input_directory, f"{name_without_ext}.wav")
            
            # Load M4A file
            print(f"Converting {base_name}...")
            audio = AudioSegment.from_file(m4a_file, format="m4a")
            
            # Export as WAV
            audio.export(wav_file, format="wav")
            print(f"Converted to {os.path.basename(wav_file)}")
            
        except Exception as e:
            print(f"Error converting {m4a_file}: {e}")
    
    print("\nConversion complete!")

# Example usage
if __name__ == "__main__":
    # Replace with your actual directory paths
    input_directory = "/home/thomwolf/Documents/Github/notebooklm-sample/"
    output_directory = input_directory
    
    # Convert all M4A files in the input directory
    convert_m4a_to_wav(input_directory, output_directory)
    
    # Alternatively, to save WAVs in the same directory:
    # convert_m4a_to_wav(input_directory)