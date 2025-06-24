# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Create Balanced Dataset (4 genres × 4 songs)
data = {
    'tempo': [90, 95, 100, 105,     # Classical
              110, 115, 120, 125,   # Jazz
              130, 135, 140, 145,   # Pop
              150, 155, 160, 165],  # Rock
    'loudness': [-20, -21, -19, -22,
                 -12, -13, -14, -11,
                 -6, -8, -10, -7,
                 -3, -4, -5, -2],
    'danceability': [0.2, 0.25, 0.3, 0.22,
                     0.4, 0.42, 0.5, 0.55,
                     0.6, 0.65, 0.7, 0.75,
                     0.8, 0.85, 0.9, 0.95],
    'genre': [
        'Classical', 'Classical', 'Classical', 'Classical',
        'Jazz', 'Jazz', 'Jazz', 'Jazz',
        'Pop', 'Pop', 'Pop', 'Pop',
        'Rock', 'Rock', 'Rock', 'Rock'
    ]
}
df = pd.DataFrame(data)

# Step 3: Encode Target (genre)
encoder = LabelEncoder()
df['genre_encoded'] = encoder.fit_transform(df['genre'])

# Step 4: Features & Labels
X = df[['tempo', 'loudness', 'danceability']]
y = df['genre_encoded']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 6: Train Multinomial Logistic Regression Model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("✅ Accuracy:", accuracy)
print("📊 Confusion Matrix:\n", cm)

# Step 8: Predict Genre for a New Song
new_song = pd.DataFrame([[135, -10, 0.7]], columns=['tempo', 'loudness', 'danceability'])
predicted_genre = model.predict(new_song)
genre_label = encoder.inverse_transform(predicted_genre)

print("🎶 Predicted Genre for new song:", genre_label[0])
