# logistic_regression

#  Intent Detection using Logistic Regression

This program is a basic example of how to detect user intent using machine learning with Python. It uses `LogisticRegression` and `TfidfVectorizer` from scikit-learn.

---

##  Step 1: Import Required Libraries

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
```

---

##  Step 2: Define Training Data

```python
X = ["book a ticket", "cancel my reservation", "what is the weather today"]
y = ["book", "cancel", "weather"]
```

- `X` is the list of input sentences (user messages).
- `y` is the list of corresponding intent labels.

---

## Step 3: Convert Text to Vectors

```python
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
```

- This converts the text data into numerical format using TF-IDF (Term Frequency–Inverse Document Frequency).

---

##  Step 4: Train the Model

```python
model = LogisticRegression()
model.fit(X_vec, y)
```

- We train a Logistic Regression model with the vectorized text and the intent labels.

---

##  Step 5: Get User Input

```python
user_input = input("Enter your message: ")
```

- Takes a new message from the user for intent prediction.

---

##  Step 6: Vectorize the User Input

```python
input_vec = vectorizer.transform([user_input])
```

- Converts the new user message to the same vector format.

---

##  Step 7: Predict Intent

```python
prediction = model.predict(input_vec)
```

- Predicts the intent using the trained model.

---

## 🖨 Step 8: Display the Detected Intent

```python
print("Intent Detected:", prediction[0])
```

- Outputs the detected intent label.

---

## Sample Output

```
Enter your message: I want to cancel my ticket
Intent Detected: cancel
```

---

##  Summary

This basic model can detect 3 intents: `book`, `cancel`, and `weather`. To improve:
- Add more training data.
- Add text cleaning (lowercasing, punctuation removal).
- Add more intents.









# Student Pass/Fail Prediction

## What it does:
Checks whether a student will pass or fail based on how many hours they studied.

## Tools Used:
- **LogisticRegression**: A machine learning model used to predict binary outcomes (pass/fail).
- **train_test_split**: Splits the data into two parts — training data to teach the model, and testing data to check accuracy.
- **accuracy_score**: Measures how accurately the model predicts the results.

## Steps:

1. Collect data for study hours and corresponding pass/fail results.
2. Separate the data into input features (study hours) and output labels (pass=1, fail=0).
3. Split the dataset into training and testing sets.
4. Train the logistic regression model using the training data.
5. Use the trained model to predict pass/fail on the test data.
6. Compare predictions with actual results to calculate accuracy.

## Example Output:

- Predicted results for test data: `[1, 0]`  
- Actual results for test data: `[1, 0]`  
- Accuracy of the model: `100%`



