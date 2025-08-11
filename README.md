# Audio Urgency Classifier with PySpark

This project focuses on detecting urgency in Italian spoken audio using machine learning models from PySpark's `pyspark.ml` library. A custom dataset was created, consisting of audio clips with and without urgency, each with multilingual transcriptions. Features were extracted from both audio (MFCC, pitch, RMSE) and text (TF-IDF), and used to train and evaluate three classifiers: Random Forest, Logistic Regression, and Gradient-Boosted Trees.

---

## Overview

- **Goal**: Classify Italian audio clips as "urgent" or "non-urgent".
- **Models**: Random Forest, Logistic Regression, Gradient-Boosted Trees (PySpark MLlib).
- **Features**: 
  - Audio: MFCCs, RMSE, pitch (via Librosa and Parselmouth)
  - Text: TF-IDF on Italian transcriptions
- **Experiments**: 
  - Audio features only
  - Text features only
  - Combined audio + text features

---

## Project Structure

The report folder contains the paper related to this project.

The src folder contains the used functions.

`requirements.txt` contains the dependencies.

The notebook folder contains the notebooks explaining the entire experimentation:

- FullSamplesTest.ipynb --> Training/testing on complete samples (Librosa + Parselmouth + TF-IDF)
- LibrosaParselmouthVStf_idf.ipynb --> Comparison of audio-only vs. text-only feature performance

The data folder contains the used samples:

- samplesCompleti.pkl --> Samples with full features (audio + text)
- samplesSoloLibrosaParselmouth.pkl --> Samples with only audio features
- samplesSoloTFIDF.pkl --> Samples with only text features

---

## How to Run the Notebooks

You can run the notebooks directly without recreating the samples from scratch.

1. Mount your Google Drive in Colab.
2. Create the following folder in your Drive:  
   `/content/drive/MyDrive/audiozzi`
3. Place the following files inside that folder:
   - `samplesCompleti.pkl`
   - `samplesSoloLibrosaParselmouth.pkl`
   - `samplesSoloTFIDF.pkl`

If you want to **rebuild the samples from raw data**, add the following to the same `audiozzi` folder:

- All `.mp3` audio files  
  - Urgent files **must end with** `u.mp3`  
- A file called `Trascrizioni.txt`, formatted as:
  
  filename1.mp3 transcript1
  
  filename2.mp3 transcript2

---

## Results Summary

- Best performance was achieved using **combined audio + text features**, with up to **100% accuracy**.
- Logistic Regression and Random Forest consistently outperformed GBT.
- Audio-only and text-only features were also effective, but slightly less accurate.

---

## License

This project is licensed under the [MIT License](./LICENSE).

The accompanying paper/report is released under the  
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

---

## Author

Created by Mattia Cocco as part of a [research project on speech-based urgency detection in Italian](https://github.com/Matti02co/BigData).  

Feel free to fork.
