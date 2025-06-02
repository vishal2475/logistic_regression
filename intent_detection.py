from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Training Data (Predefined sample messages and intents)
X = ["book a ticket", "cancel my reservation", "what is the weather today"]
y = ["book", "cancel", "weather"]

# 2. Convert text to numerical vectors
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# 3. Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_vec, y)

# 4. Take user input
user_input = input("Enter your message: ")

# 5. Convert user input to vector
input_vec = vectorizer.transform([user_input])

# 6. Predict the intent
prediction = model.predict(input_vec)

# 7. Print the result
print("Intent Detected:", prediction[0])
