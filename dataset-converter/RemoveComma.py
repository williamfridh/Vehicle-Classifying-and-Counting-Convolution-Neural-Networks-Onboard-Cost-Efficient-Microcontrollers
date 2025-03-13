import os
import re

folder_path = r"C:\Users\Pontu\OneDrive\Skrivbord\lib3\Vehicle-Classifying-and-Counting-Convolution-Neural-Networks-Onboard-Cost-Efficient-Microcontrollers\dataset-converter\output_frames\5"

toChange = "Truck"
index = "5"

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # Process only .txt files
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as file:
            content = file.read()

        # Replace [[ with [[[, but not if it's already [[[.
        #content = re.sub(r'(?<!\[)\[\[', '[[[', content)
        
        content = content.replace(toChange, index)

        #if content.startswith("[[[["):
        #    content = content[1:]  # Remove first element of the list 
        
        # Remove the last ", " if it exists
        #if content.endswith(", "):
        #    content = content[:-2]  # Remove last character if it's a comma
        
        # Write the modified content back to the file
        with open(file_path, "w") as file:
            file.write(content)
