import pandas as pd
import os
import shutil

# Load CSV
data = pd.read_csv('esc50.csv')

print(data.head())

# Define base paths
source_base = r"C:\Users\Pontu\OneDrive\Skrivbord\lib3\Vehicle-Classifying-and-Counting-Convolution-Neural-Networks-Onboard-Cost-Efficient-Microcontrollers\Dataset\audio"
dest_base = r"C:\Users\Pontu\OneDrive\Skrivbord\lib3\Vehicle-Classifying-and-Counting-Convolution-Neural-Networks-Onboard-Cost-Efficient-Microcontrollers\Dataset\Dataset"

# Define destination folder
dest_label_folder = os.path.join(dest_base, 'Background_noise')

# Ensure the destination folder exists
os.makedirs(dest_label_folder, exist_ok=True)

# Iterate over rows and move matching files
for index, row in data.iterrows():
    filename = row['filename']  # No need to add 'Y' unless filenames actually start with 'Y'

    # Define source and destination paths
    src_path = os.path.join(source_base, filename)
    dst_path = os.path.join(dest_label_folder, filename)

    # Move file if it exists
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"Moved: {src_path} → {dst_path}")
    else:
        print(f"File not found: {src_path}")
