
import pandas as pd
import os
import shutil



#1. Check csv, and the label

#2. Take file from the C:/Users/Pontu/OneDrive/Skrivbord/Dataset/unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads folder

#3. Put the file into the correct labeled folder 

# Change csv depending on 

csv_file = 'esc50.csv'

# Change the category
label = 'door_wood_creaks'

source = 'C:/Users/Pontu/OneDrive/Skrivbord/lib/Vehicle-Classifying-and-Counting-Convolution-Neural-Networks-Onboard-Cost-Efficient-Microcontrollers/Dataset/audio'

dest = os.path.join('C:/Users/Pontu/OneDrive/Skrivbord/lib/Vehicle-Classifying-and-Counting-Convolution-Neural-Networks-Onboard-Cost-Efficient-Microcontrollers/Dataset/split_up_data/background_noise/'
                    , label)

data = pd.read_csv(csv_file)

for index, row in data.iterrows():
    if row['category'] == label:
        # Take the audio from source and put it in dest
        filename = row['filename']
        src_path = os.path.join(source, filename)
        dst_path = os.path.join(dest, filename)

        
        if os.path.exists(src_path):  # Ensure source file exists
            shutil.move(src_path, dst_path)
            print(f"Moved: {src_path} → {dst_path}")
        else:
            print(f"File not found: {src_path}")