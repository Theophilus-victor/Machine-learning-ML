import librosa
import N as np
import os
import pandas as pd

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    # Extracting MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Extracting Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    
    # Combine all features into one array
    return np.hstack([mfcc_mean, chroma_mean, spectral_contrast_mean])

# Path to the dataset folder
dataset_path = "path_to_audio_dataset"  # Modify this path
genres = os.listdir(dataset_path)
features = []
labels = []

for genre in genres:
    genre_folder = os.path.join(dataset_path, genre)
    if os.path.isdir(genre_folder):
        for file_name in os.listdir(genre_folder):
            if file_name.endswith(".wav"):
                audio_path = os.path.join(genre_folder, file_name)
                feature_vector = extract_features(audio_path)
                features.append(feature_vector)
                labels.append(genre)

# Save the data to a pandas dataframe
df = pd.DataFrame(features)
df['label'] = labels

# Save the dataframe to CSV
df.to_csv('music_genre_data.csv', index=False)
