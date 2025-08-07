# Module for feature extraction functions

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call


def extract_audio_features(audio_path, sr=22050, n_mfcc=13):
    """
    Extracts MFCC and RMSE audio features from the given file.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    sr : int, optional
        Sampling rate (default is 22050).
    n_mfcc : int, optional
        Number of MFCCs to extract (default is 13).

    Returns
    -------
    np.ndarray
        Concatenated array of MFCC means and RMSE mean.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    return np.concatenate((mfccs_mean, [rms_mean]))


def extract_pitch(audio_path):
    """
    Extracts mean pitch (F0) from the audio file using Parselmouth.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.

    Returns
    -------
    np.ndarray
        Array containing mean pitch value in Hertz.
    """
    snd = parselmouth.Sound(audio_path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    return np.array([mean_pitch])


def extract_text_features(text, vectorizer):
    """
    Transforms text into TF-IDF feature vector.

    Parameters
    ----------
    text : str
        Input text to transform.
    vectorizer : TfidfVectorizer
        Pre-fitted TF-IDF vectorizer.

    Returns
    -------
    np.ndarray
        Transformed TF-IDF feature vector.
    """
    return vectorizer.transform([text]).toarray()[0]


def load_transcriptions(txt_file):
    """
    Loads transcriptions from a .txt file and stores them in a dictionary.

    Parameters
    ----------
    txt_file : str
        Path to the transcription text file.

    Returns
    -------
    dict
        Dictionary mapping audio file names to transcriptions.
    """
    transcriptions = {}
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n\n")
    for block in lines:
        lines = block.split("\n")
        if len(lines) >= 2:
            filename = lines[0].strip()
            transcript = " ".join(lines[1:]).strip()
            transcriptions[filename] = transcript
    return transcriptions

