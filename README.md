# Song Popularity Prediction

## Overview

This project aims to predict the popularity of songs based on various features such as acousticness, danceability, energy, instrumentalness, loudness, speechiness, tempo, valence, etc. The prediction is done using machine learning techniques implemented in Python.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#dataset)
- [Features](#features)
- [License](#license)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Infinibyte-Github/SpotifyPopularityPrediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd SpotifyPopularityPrediction
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Preprocess the data:

    ```bash
    python 1_1_Preprocessing.py
    ```

2. Split the data:

    ```bash
    python 2_Splitting.py
    ```

3. Scale the data:

    ```bash
    python 3_Scaling.py
    ```

## Dataset

I used the "🎹 Spotify Tracks Dataset" by *MaharshiPandya* on Kaggle ([Link](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)). However, you can use any suitable dataset containing song features and popularity ratings.

## Features

The features used for predicting song popularity are:

| Feature          | Data Type | Description                                                                                                                                                                                                                                                                    |
|------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| duration_ms      | int64     | The track length in milliseconds.                                                                                                                                                                                                                                              |
| explicit         | bool      | Whether or not the track has explicit lyrics. "true" indicates the track has explicit lyrics, "false" indicates the track does not have explicit lyrics, or "unknown" if the information is not available.                                                                     |
| danceability     | float64   | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable, and 1.0 is most danceable.                                  |
| energy           | float64   | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale.                             |
| key              | int64     | The key the track is in, mapped to pitches using standard Pitch Class notation. Integers map to pitches as follows: 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.                                                                               |
| loudness         | float64   | The overall loudness of a track in decibels (dB).                                                                                                                                                                                                                              |
| mode             | int64     | Mode indicates the modality (major or minor) of a track, represented as 1 for major and 0 for minor.                                                                                                                                                                           |
| speechiness      | float64   | Speechiness detects the presence of spoken words in a track. The attribute value ranges from 0.0 to 1.0, where values closer to 1.0 describe tracks that are more speech-like (e.g., talk show, audio book) and values closer to 0.0 describe tracks that are more music-like. |
| acousticness     | float64   | A confidence measure from 0.0 to 1.0 indicating whether the track is acoustic. A value of 1.0 represents high confidence that the track is acoustic.                                                                                                                           |
| instrumentalness | float64   | Predicts whether a track contains no vocals. Values closer to 1.0 indicate a higher likelihood that the track contains no vocal content.                                                                                                                                       |
| liveness         | float64   | Liveness detects the presence of an audience in the recording. A value above 0.8 indicates a strong likelihood that the track was performed live.                                                                                                                              |
| valence          | float64   | Valence is a measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Higher values represent a more positive mood (e.g., happy, cheerful), while lower values represent a more negative mood (e.g., sad, depressed).                                  |
| tempo            | float64   | The overall estimated tempo of a track in beats per minute (BPM).                                                                                                                                                                                                              |
| track_genre      | object    | The genre(s) in which the track belongs.                                                                                                                                                                                                                                       |


## License

This project is licensed under the GNU GPL License - see the [LICENSE](LICENSE) file for details.
