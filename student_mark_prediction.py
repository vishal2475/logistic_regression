import pandas as pd
from mord import LogisticAT
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Dataset
data = {
    'students': ['gokul', 'vishal', 'agash', 'navnith', 'akash', 'praveen', 'ayush', 'amar', 'arun', 'muthu'],
    'Total_T1': [260, 410, 285, 148, 450, 345, 291, 125, 178, 440],
    'Total_T2': [280, 418, 306, 168, 462, 368, 303, 141, 195, 453]
}
df = pd.DataFrame(data)

# Step 2: Estimate Term 3 total
df['Estimated_Total_T3'] = ((df['Total_T1'] + df['Total_T2']) / 2).round()

# Step 3: Convert to category
def to_category(m):
    if m < 200:
        return 0  # Fail
    elif m < 300:
        return 1  # Low Pass
    elif m < 400:
        return 2  # Good Pass
    else:
        return 3  # High Pass

df['T3_Category'] = df['Estimated_Total_T3'].apply(to_category)

# Step 4: Train ordinal logistic regression model
X = df[['Total_T1', 'Total_T2']]
y = df['T3_Category']

model = LogisticAT()
model.fit(X, y)

# Step 5: Predict
pred = model.predict(X)

# Step 6: Map predictions to labels
labels = ['Fail', 'Low Pass', 'Good Pass', 'High Pass']
df['Predicted_Label'] = [labels[i] for i in pred]

# ✅ Step 7: Print final results
print("\n🎯 Final Prediction Result:\n")
print(df[['students', 'Total_T1', 'Total_T2', 'Estimated_Total_T3', 'T3_Category', 'Predicted_Label']])

# Optional: print accuracy (optional since we're using same data for training)
from sklearn.metrics import accuracy_score
print("\n✅ Accuracy:", accuracy_score(y, pred))
