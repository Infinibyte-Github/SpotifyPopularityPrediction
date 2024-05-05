"""
Scaling of dataset for training purposes
Author: Bart Schelpe
Filename: 3_Scaling.py
Dataset: 🎹 Spotify Tracks Dataset
Link: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
Code based on: Python4AI PowerPoint presentations and documentation of varying Python packages
"""

# import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# read dataframes from pickle files
dfTrain = pd.read_pickle("../data/dfTrain.pickle")
dfTest = pd.read_pickle("../data/dfTest.pickle")

# TODO: fix the OneHotEncoder
# # apply one-hot encoding to the track_genre column
# categories = dfTrain['track_genre'].unique().reshape(-1, 1)
# categories = [categories.flatten().tolist()]
# enc = OneHotEncoder(categories=categories)
# enc.fit(dfTrain[['track_genre']])
#
# # transform the dataframes
# dfTrain = pd.concat([dfTrain, pd.DataFrame(enc.transform(dfTrain[['track_genre']]).toarray())], axis=1)
# dfTest = pd.concat([dfTest, pd.DataFrame(enc.transform(dfTest[['track_genre']]).toarray())], axis=1)

# drop the original track_genre column
dfTrain = dfTrain.drop(columns=['track_genre'])
dfTest = dfTest.drop(columns=['track_genre'])

# Define the number of keys (12 for 12 musical pitches)
num_keys = 12

# Encode the 'key' feature using sine-cosine encoding
dfTrain['key_sin'] = np.sin(2 * np.pi * dfTrain['key'] / num_keys)
dfTrain['key_cos'] = np.cos(2 * np.pi * dfTrain['key'] / num_keys)

dfTest['key_sin'] = np.sin(2 * np.pi * dfTest['key'] / num_keys)
dfTest['key_cos'] = np.cos(2 * np.pi * dfTest['key'] / num_keys)

# Drop the original 'key' columns
dfTrain.drop(columns=['key'], inplace=True)
dfTest.drop(columns=['key'], inplace=True)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the training data
scaler.fit_transform(dfTrain)

# Transform the dataframes
dfTrain[dfTrain.columns] = scaler.transform(dfTrain[dfTrain.columns])
dfTest[dfTest.columns] = scaler.transform(dfTest[dfTest.columns])

# Save the dataframes to new pickle files
dfTrain.to_pickle("../data/dfTrainMinMaxScaler.pickle")
dfTest.to_pickle("../data/dfTestMinMaxScaler.pickle")
