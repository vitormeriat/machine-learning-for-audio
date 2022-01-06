# Load various imports
import os
import pandas as pd
from tqdm import tqdm
import helper as helper


if os.path.isfile('./data/features_df.pkl'):
    print("[INFO] File exist")
else:
    # Set the path to the full UrbanSound dataset
    #fulldatasetpath = '/Urban Sound/UrbanSound8K/audio/'
    fulldatasetpath = 'C:/research/mvpconf/urbansound8k/'

    metadata = helper.create_metadata_df(fulldatasetpath + 'UrbanSound8K.csv')
    print(metadata.head())
    print('\n[INFO] Start feature extraction\n')

    metadata_list = helper.create_metadata_list(metadata)

    features = []
    # Iterate through each sound file and extract the features
    for i in tqdm(metadata_list):
        class_label = i[0]
        data = helper.extract_features(i[1])
        features.append([data, class_label])
    #features = helper.extract_features(metadata_list)

    # Convert into a Panda dataframe
    features_df = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('\n[INFO] Finished feature extraction from ',
          len(features_df), ' files')

    #pickle the dataframe
    features_df.to_pickle('./data/features_df.pkl')

    print('[INFO] Pickle a DataFrame...')
