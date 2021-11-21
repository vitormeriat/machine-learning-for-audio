# Load various imports
import os
#import pickle
#from librosa import feature
import pandas as pd
from tqdm import tqdm
#from helper import extract_features
import helper as helper


# if os.path.exists('../data/features_df'):
if os.path.isfile('./data/features_df.pkl'):
    print("[INFO] File exist")
else:
    # Set the path to the full UrbanSound dataset
    #fulldatasetpath = '/Urban Sound/UrbanSound8K/audio/'
    fulldatasetpath = 'C:/research/mvpconf/urbansound8k/'

    #metadata = pd.read_csv(fulldatasetpath + '../metadata/UrbanSound8K.csv')
    #metadata = pd.read_csv(fulldatasetpath + 'UrbanSound8K.csv')
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

    # with open('../data/features_df', 'wb') as picklefile:
    #    #pickle the dataframe
    #    pickle.dump(features_df, picklefile)
    features_df.to_pickle('./data/features_df.pkl')

    print('[INFO] Pickle a DataFrame...')
