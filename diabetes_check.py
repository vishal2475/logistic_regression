import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset (Glucose, BMI, Age, Outcome)
data = {
    'Glucose': [148, 85, 183, 89, 137, 116],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6],
    'Age': [50, 31, 32, 21, 33, 30],
    'Outcome': [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Glucose', 'BMI', 'Age']]
y = df['Outcome']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print results
print("Predicted:", y_pred)
print("Actual   :", y_test.values)
print("Accuracy :", accuracy_score(y_test, y_pred))
