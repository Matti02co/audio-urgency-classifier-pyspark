import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.audio_features import extract_audio_features, extract_pitch, extract_text_features, load_transcriptions

# Define paths
audio_folder = "/content/drive/MyDrive/audiozzi"
transcriptions_file = os.path.join(audio_folder, "Trascrizioni.txt")

# Load transcriptions into dictionary
transcriptions = load_transcriptions(transcriptions_file)

# Create TF-IDF vectorizer and fit it on the corpus
vectorizer = TfidfVectorizer()
vectorizer.fit(list(transcriptions.values()))

# Initialize dataset list
dataset = []

# Iterate over audio files
for file in os.listdir(audio_folder):
    if file.endswith(".mp3"):
        audio_path = os.path.join(audio_folder, file)

        # Assign label based on filename convention
        label = 1 if file.endswith("u.mp3") else 0

        # Retrieve corresponding transcription
        transcript = transcriptions.get(file, None)
        if transcript is None:
            print(f"No transcription for {file}, skipped.")
            continue

        # Extract features
        audio_features = extract_audio_features(audio_path)
        pitch_feature = extract_pitch(audio_path)
        text_features = extract_text_features(transcript, vectorizer)

        # Concatenate all features
        sample = np.concatenate((audio_features, pitch_feature, text_features))

        # Save as dictionary
        dataset.append({
            "filename": file,
            "features": sample,
            "transcript": transcript,
            "label": label
        })

# Convert to DataFrame and save to binary file
df = pd.DataFrame(dataset)
df.to_pickle(os.path.join(audio_folder, "samplesCompleti.pkl"))
print(f"{len(df)} saved samples.")

