"""
Splitting of dataset for training purposes
Author: Bart Schelpe
Filename: 2_Splitting.py
Dataset: 🎹 Spotify Tracks Dataset
Link: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
Code based on: Python4AI PowerPoint presentations and documentation of varying Python packages
"""

# import packages
import pandas as pd

# read dataframe from pickle file
df = pd.read_pickle("../data/df.pickle")

# shuffle the dataset
df = df.sample(frac=1, random_state=69)

print(df.head())

# Split the shuffled DataFrame into training and testing sets
train_size = int(0.8 * len(df))  # 80% for training
train_df = df[:train_size]
test_df = df[train_size:]

# save both dataframes to new pickle files
train_df.to_pickle("../data/dfTrain.pickle")
test_df.to_pickle("../data/dfTest.pickle")
