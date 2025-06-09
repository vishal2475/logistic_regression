import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6],
    'Passed': [0, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# Features and label
X = df[['StudyHours']]
y = df['Passed']

# Split data (67% training, 33% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
print("Actual:", y_test.values)
print("Predicted:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
