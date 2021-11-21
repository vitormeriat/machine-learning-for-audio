from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np


def convert(df):
    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(df.feature.tolist())
    y = np.array(df.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y)) 

    # split the dataset 
    from sklearn.model_selection import train_test_split 

    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
    return x_train, x_test, y_train, y_test
