# Random Sound Recognition

A 1D CNN that classifies environmental sounds. Training extracts MFCC features from the UrbanSound8K clips (the CSV file name and columns suggest that dataset) and trains a Keras model. The live test script records 5-second chunks from the microphone and prints the predicted sound class.

## Features

- MFCC feature extraction with librosa (50 coefficients, averaged over time, 22,050 Hz).
- Conv1D classifier with three convolution blocks and two dense layers, trained for 50 epochs with an 80/20 split.
- Real-time prediction from the default microphone, one prediction per 5-second chunk.

## Project structure

- `train.py` - extracts features from `Data/`, trains the model, saves `sound_classification_model.keras` and `label_encoder.pkl`.
- `test.py` - records from the microphone and prints predicted classes.
- `sound_classification_model.keras` - trained model used by `test.py`.
- `label_encoder.pkl` - label encoder saved by `train.py`.
- `DEMO.mp4` - demo video.
- `Readme.txt` - original install and run notes.
- `Data/` - not included (see Notes). Expected to hold `UrbanSound8K.csv` and `fold1` to `fold10` audio folders.

## Requirements

Imports seen in the code: `numpy`, `pandas`, `librosa`, `sklearn`, `tensorflow`, `joblib`, `sounddevice`.

```bash
pip install numpy pandas librosa scikit-learn tensorflow joblib sounddevice
```

## Installation

```bash
git clone <this-repo-url>
cd Random-Sound-Recognition
pip install numpy pandas librosa scikit-learn tensorflow joblib sounddevice
```

`sounddevice` needs the PortAudio library; on Windows the wheel includes it.

## Usage

```bash
# Optional: retrain. Requires the Data/ folder (see Notes). Overwrites the .keras and .pkl files.
python train.py

# Live prediction from the microphone. Press Ctrl+C to stop.
python test.py
```

`test.py` loads `sound_classification_model.keras` and `label_encoder.pkl` from the current folder, so run it from this folder.

## Notes

- **Excluded: `Data/` folder** (about 6.7 GB of audio). `train.py` expects:
  - `Data/UrbanSound8K.csv`, with columns `slice_file_name, fsID, start, end, salience, fold, classID, class`.
  - `Data/fold1/` to `Data/fold10/`, each containing the WAV files named in the CSV's `slice_file_name` column (for example `Data/fold5/100032-3-0-0.wav`).
- `train.py` uses hard-coded absolute paths under `D:/PythonProjects/Random Sound Recognition/Data`. Change `pd.read_csv` and `path` to point to your `Data/` folder before retraining.
- `train.py` extracts features for every row in the CSV, so it is slow without caching.
- `test.py` expects a microphone and uses 5-second chunks; sound is not classified continuously between chunks.
- The model was not evaluated in this README. Accuracy numbers are not documented in the code.
- The `DEMO.mp4` file shows the live test in use.
