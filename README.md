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


