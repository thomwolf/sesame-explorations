import os
import glob

def concatenate_rttm_files(input_folder, output_file):
    """
    Concatenates all .rttm files in the input folder into a single output file.
    
    Args:
        input_folder (str): Path to the folder containing .rttm files
        output_file (str): Path to the output file
    """
    # Get all .rttm files in the folder
    rttm_files = glob.glob(os.path.join(input_folder, "*.rttm"))
    
    if not rttm_files:
        print(f"No .rttm files found in {input_folder}")
        return
    
    # Open the output file for writing
    with open(output_file, 'w') as outfile:
        # Process each input file
        for rttm_file in sorted(rttm_files):
            print(f"Processing {rttm_file}")
            
            # Open the input file and read its content
            with open(rttm_file, 'r') as infile:
                # Write the content to the output file
                outfile.write(infile.read())
    
    print(f"Concatenation complete. Output written to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Concatenate all .rttm files in a folder")
    parser.add_argument("input_folder", help="Path to the folder containing .rttm files")
    parser.add_argument("output_file", help="Path to the output file")
    
    args = parser.parse_args()
    
    concatenate_rttm_files(args.input_folder, args.output_file)

# Example usage
# python concatenate_rttm_files.py /path/to/your/folder output.rttm