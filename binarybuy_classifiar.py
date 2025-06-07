import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample Data
data = {
    'Age': [18, 20, 22, 25, 28, 30, 35, 40, 45, 50],
    'Buy': [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Labels
X = df[['Age']]  # Input feature (Age)
y = df['Buy']    # Output label (Buy or Not)

# Create and Train the Model
model = LogisticRegression()
model.fit(X, y)

# Predict for new ages
test_ages = [[19], [29], [37], [49]]
predictions = model.predict(test_ages)

# Show the results
for age, result in zip(test_ages, predictions):
    status = "Buy" if result == 1 else "Do Not Buy"
    print(f"Age {age[0]} => {status}")
