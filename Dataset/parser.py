
import numpy as np
import pandas as pd
import os
import shutil



#1. Check csv, and the label

#2. Take file from the C:\Users\Pontu\OneDrive\Skrivbord\Dataset\unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads folder

#3. Put the file into the correct labeled folder 



data = pd.read_csv('groundtruth_weak_label_training_set.csv')

print(data.head())


for index, row in data.iterrows():
    # Change the label depending on what to 
    if(row['label'] == 'Bus'):
        # Take the audio from source and put it in dest
        filename = 'Y' + row['filename']
        source = 'C:/Users/Pontu/OneDrive/Skrivbord/Dataset/unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads/'
        destLabel = 'Bus'
        dest = 'C:/Users/Pontu/OneDrive/Skrivbord/Dataset/' + destLabel

        src_path = os.path.join(source, filename)
        dst_path = os.path.join(dest, filename)

        if os.path.exists(src_path):  # Ensure source file exists
            shutil.move(src_path, dst_path)
            print(f"Moved: {src_path} → {dst_path}")
        else:
            print(f"File not found: {src_path}")