# Load various imports
import os
import pandas as pd
from helper import extract_features


# Set the path to the full UrbanSound dataset
fulldatasetpath = '/Urban Sound/UrbanSound8K/audio/'

metadata = pd.read_csv(fulldatasetpath + '../metadata/UrbanSound8K.csv')

features = []

# Iterate through each sound file and extract the features
for index, row in metadata.iterrows():

    file_name = os.path.join(os.path.abspath(
        fulldatasetpath), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))

    class_label = row["class_name"]
    data = extract_features(file_name)

    features.append([data, class_label])

# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')
