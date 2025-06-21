# Full Project: Student Third-Term Performance Prediction using Ordinal Logistic Regression

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mord import LogisticAT  # Ordinal Logistic Regression
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Create Dataset
data = {
    'Student': ['gokul', 'vishal', 'agash', 'navnith', 'akash', 'praveen', 'ayush', 'amar', 'arun', 'muthu'],
    'Total_T1': [260, 410, 285, 300, 390, 240, 350, 310, 230, 200],
    'Total_T2': [280, 420, 295, 330, 395, 250, 360, 330, 250, 210],
    'Total_T3': [290, 430, 310, 340, 400, 270, 370, 350, 260, 220]
}

df = pd.DataFrame(data)

# Step 2: Categorize Total_T3 into Ordinal Labels
def get_category(mark):
    if mark < 270:
        return 'Low'
    elif 270 <= mark <= 370:
        return 'Medium'
    else:
        return 'High'

df['T3_Category'] = df['Total_T3'].apply(get_category)

# Step 3: Encode Ordinal Labels (Low=0, Medium=1, High=2)
label_encoder = LabelEncoder()
df['T3_Label'] = label_encoder.fit_transform(df['T3_Category'])

# Step 4: Split Data into Features and Target
X = df[['Total_T1', 'Total_T2']]
y = df['T3_Label']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train Ordinal Logistic Regression Model
model = LogisticAT()
model.fit(X_train, y_train)

# Step 7: Predict on Test Data
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("\n===== Evaluation Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# Step 9: Convert predictions back to category
y_test_cat = label_encoder.inverse_transform(y_test)
y_pred_cat = label_encoder.inverse_transform(y_pred)

# Step 10: Create Comparison DataFrame
compare_df = pd.DataFrame({
    'Student': df.loc[y_test.index, 'Student'],
    'Actual': y_test_cat,
    'Predicted': y_pred_cat
})

print("\n===== Actual vs Predicted =====")
print(compare_df)

# Step 11: Visualize Actual vs Predicted Categories
plt.figure(figsize=(8, 5))
sns.countplot(x='Actual', data=compare_df, palette='Blues', label='Actual')
sns.countplot(x='Predicted', data=compare_df, palette='Oranges', label='Predicted', alpha=0.6)
plt.title('Actual vs Predicted T3 Category')
plt.legend()
plt.show()
