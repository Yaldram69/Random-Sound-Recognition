# Basic Libraries
import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
df = pd.read_csv("D:/PythonProjects/Random Sound Recognition/Data/UrbanSound8K.csv")

# Extract features from audio files
def features_extract(file):
    sample, sample_rate = librosa.load(file, sr=22050)  # Fixed sample rate
    feature = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=50)
    scaled_feature = np.mean(feature.T, axis=0)
    return scaled_feature

extracted = []
path = 'D:/PythonProjects/Random Sound Recognition/Data'

for index_num, row in df.iterrows():
    file_name = os.path.join(path, 'fold' + str(row["fold"]), str(row['slice_file_name']))
    final_class_labels = row['class']
    data = features_extract(file_name)
    extracted.append([data, final_class_labels])

ext_df = pd.DataFrame(extracted, columns=['feature', 'class'])

x = np.array(ext_df['feature'].tolist())
y = np.array(ext_df['class'].tolist())

# Encode labels
le = LabelEncoder()
y = to_categorical(le.fit_transform(y))

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Reshape for Conv1D (add channel dimension)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Model Initialization
model = Sequential()
model.add(Conv1D(64, kernel_size=3, padding='same', input_shape=(50, 1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, kernel_size=3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(256, kernel_size=3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(len(le.classes_)))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    x_train, y_train,
    batch_size=16,
    epochs=50,  # Adjust as needed
    validation_data=(x_test, y_test),
)

# Save the model and label encoder
model.save('sound_classification_model.keras')
joblib.dump(le, 'label_encoder.pkl')
