"""
Preprocessing of dataset
Author: Bart Schelpe
Filename: 1_Preprocessing.py
Dataset: 🎹 Spotify Tracks Dataset
Link: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
Code based on: Python4AI PowerPoint presentations and documentation of varying Python packages
"""

# import packages
import pandas as pd

# read data from csv file
print('--------------------')
print('Loading Spotify Dataset...')
df = pd.read_csv("../data/dataset.csv")
print('Spotify Dataset loaded!')

# drop the columns that are not needed for the model
print('--------------------')
print('Preprocessing Dataset...')
not_needed = ["Unnamed: 0", "track_id", "artists", "album_name", "track_name", "time_signature"]
df = df.drop(columns=not_needed, axis=1)
# df.info()

# drop incomplete rows
df = df.dropna()
print('Preprocessing Dataset done!')

# save the dataframe to a pickle file
print('--------------------')
print('Saving Spotify Dataset to pickle file...')
df.to_pickle("../data/df.pickle")
print('Spotify Dataset saved to pickle file!')
print('--------------------')
