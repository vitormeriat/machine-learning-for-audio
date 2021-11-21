import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def create_metadata_df(csv_file):
    metadata = pd.read_csv(csv_file)
    fulldatasetpath = 'C:/research/mvpconf/urbansound8k/'
    filepaths = [
        os.path.join(
            fulldatasetpath,
            'fold' + str(row['fold']),
            row['slice_file_name'],
        )
        for _, row in metadata.iterrows()
    ]

    metadata['filepath'] = filepaths
    return metadata


def create_metadata_list(df):
    return [
        [row['class'], row['filepath']] for _, row in df.iterrows()
    ]


def extract_features(file_name):

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs_scaled


# def parser(metadata_list):
#     # Feature Extraction and mel_spectogram function of librosa to extract the spectogram data as a numpy array
#     feature = []
#     label = []

#     # Function to load files and extract features
#     for i in tqdm(metadata_list):
#         # for i in range(1,8732):
#         #file_name = 'urbansound8K/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
#         # Here kaiser_fast is a technique used for faster extraction
#         X, sample_rate = librosa.load(i[1], res_type='kaiser_fast')
#         # We extract mfcc feature from data , first extracting 3s per file....ie 22050*0.5=11025
#         # applying segmentation:
#         segment_1s = [X[k*22050: k*22050+22050]
#                       for k in range(int(len(X)/22050))]
#         for audio_1s in segment_1s:
#             mels = librosa.feature.melspectrogram(y=audio_1s, sr=sample_rate).T
#             feature.append(mels)
#             label.append(i[0])

#     return [feature, label]


# def extract_features(metadata_list):
#     temp = parser(metadata_list)
#     temp = np.array(temp, dtype=object)
#     return temp.transpose()
