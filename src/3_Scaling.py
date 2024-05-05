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

# read dataframe from pickle file
df = pd.read_pickle("../data/df.pickle")

# print column names and the data types
print(df.dtypes)
