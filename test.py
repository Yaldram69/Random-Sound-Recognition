import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import joblib
import time
import sys

# Load the saved model and label encoder
model = load_model('sound_classification_model.keras')
le = joblib.load('label_encoder.pkl')

# Parameters for real-time audio capture
SAMPLE_RATE = 22050
DURATION = 5  # Duration in seconds for each audio chunk
N_MFCC = 50  # Number of MFCC features
CHUNK_SIZE = SAMPLE_RATE * DURATION

def extract_feature(audio_data):
    # Ensure audio_data is a 1D array
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    fea = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    scaled = np.mean(fea.T, axis=0)
    return np.array([scaled])

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)

    if len(indata) >= CHUNK_SIZE:
        audio_data = indata[:CHUNK_SIZE].flatten()  # Flatten to ensure 1D array
        feature = extract_feature(audio_data)
        prediction = model.predict(feature, batch_size=1)
        pred_vector = np.argmax(prediction, axis=-1)
        pred_class = le.inverse_transform(pred_vector)
        print(f"Predicted class: {pred_class[0]}")

# Set up the audio stream
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(DURATION)
    except KeyboardInterrupt:
        print("Stopped.")
