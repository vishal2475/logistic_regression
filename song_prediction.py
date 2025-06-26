import os
import librosa
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Step 1: Feature Extraction Function
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = mfccs.mean(axis=1)
    features = [tempo, centroid, zcr] + list(mfccs_mean)
    return features

# Step 2: Extract features and prepare labeled dataset manually
def prepare_dataset(mp3_folder):
    rows = []
    print("Please enter the genre for each song when prompted:\n")
    for filename in os.listdir(mp3_folder):
        if filename.endswith(".mp3"):
            file_path = os.path.join(mp3_folder, filename)
            features = extract_features(file_path)
            print(f"Enter genre for {filename} (e.g. Pop, Rock, Classical, etc.): ", end="")
            genre = input()
            features.append(genre)
            rows.append(features)

    columns = ['tempo', 'centroid', 'zcr'] + [f'mfcc{i+1}' for i in range(13)] + ['genre']
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv("music_features.csv", index=False)
    print("\nDataset saved as 'music_features.csv'\n")

# Step 3: Train Model from CSV
def train_model():
    data = pd.read_csv("music_features.csv")
    X = data.drop("genre", axis=1)
    y = data["genre"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, "model.pkl")
    print("\nModel saved as 'model.pkl'\n")

# Step 4: Predict a new song
def predict_song(file_path):
    features = extract_features(file_path)
    model = joblib.load("model.pkl")
    prediction = model.predict([features])
    return prediction[0]

