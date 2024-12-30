#!/usr/bin/env python
# coding: utf-8

# ## INTERN:-VISHAL RAMKUMAR RAJBHAR
# Intern ID:- CT4MESG
# 
# Domain:- Machine Learning
# 
# Duration:-December 17, 2024, to April 17, 2025
# 
# Company:- CODETECH IT SOLUTIONS
# 
# Mentor:- Neela Santhosh Kumar

# ## TASK TWO: SENTIMENT ANALYSIS WITH NLP
# 
# PERFORM SENTIMENT ANALYSIS ON A DATASET OF CUSTOMER REVIEWS USING TF-IDF VECTORIZATION AND LOGISTIC REGRESSION.
# DELIVERABLE: A JUPYTER NOTEBOOK SHOWCASING PREPROCESSING, MODELING,AND SENTIMENT EVALUATION

# In[2]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# Creating a simple dataset with customer reviews and sentiment labels
# 1 = Positive sentiment, 0 = Negative sentiment
data = pd.DataFrame({
    "review": [
        "This product is amazing!", 
        "Terrible service, would not recommend.",
        "I love it, works perfectly.",
        "Not worth the money. Completely dissatisfied.",
        "Very good quality and fast delivery.",
        "Awful experience. The item arrived broken.",
        "Exceptional! Exceeded my expectations.",
        "Mediocre at best. Expected better.",
        "Fantastic! Will buy again.",
        "Poor customer service. Never buying here again."
    ],
    "sentiment": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Sentiment labels
})

# Display dataset preview
print("Dataset preview:")
print(data.head())

# Step 2: Split the dataset into features (X) and labels (y)
X = data['review']
y = data['sentiment']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorize the text data using TF-IDF
# Transform the text data into numerical format for modeling
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Initialize the Logistic Regression model
model = LogisticRegression(random_state=42)

# Step 6: Train the model on the training data
model.fit(X_train_tfidf, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluate the model
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Step 9: Plot the Confusion Matrix
# Visualize the confusion matrix to better understand predictions
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 10: Example predictions on new reviews
# Input some example reviews to test the model
example_reviews = [
    "The product quality is excellent!",
    "Horrible experience, will not return.",
    "Decent product but expected better.",
    "Absolutely love it! Fast delivery too."
]

# Transform the example reviews using the TF-IDF vectorizer
example_tfidf = vectorizer.transform(example_reviews)

# Predict the sentiment for each example review
example_preds = model.predict(example_tfidf)

# Display predictions
for review, sentiment in zip(example_reviews, example_preds):
    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    print(f"Review: '{review}' | Sentiment: {sentiment_label}")

# Step 11: Visualize Example Predictions
# Create a bar plot to visualize the sentiment predictions
sentiments = ["Positive" if s == 1 else "Negative" for s in example_preds]
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiments, y=[1]*len(sentiments), palette="viridis", dodge=False)
plt.title("Sentiment Predictions for Example Reviews")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Count")
plt.show()


# In[4]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


# Step 1: Load the dataset
# Creating a simple dataset with customer reviews and sentiment labels
# 1 = Positive sentiment, 0 = Negative sentiment
data = pd.DataFrame({
    "review": [
        "This product is amazing!", 
        "Terrible service, would not recommend.",
        "I love it, works perfectly.",
        "Not worth the money. Completely dissatisfied.",
        "Very good quality and fast delivery.",
        "Awful experience. The item arrived broken.",
        "Exceptional! Exceeded my expectations.",
        "Mediocre at best. Expected better.",
        "Fantastic! Will buy again.",
        "Poor customer service. Never buying here again."
    ],
    "sentiment": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Sentiment labels
})

# Display dataset preview
print("Dataset preview:")
print(data.head(0))


# In[8]:


print(data.head(5))
print(data.tail(5))


# In[13]:


# Step 2: Split the dataset into features (X) and labels (y)
X = data['review']
y = data['sentiment']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorize the text data using TF-IDF
# Transform the text data into numerical format for modeling
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# In[18]:


# Step 5: Initialize the Logistic Regression model
model = LogisticRegression(random_state=42)

# Step 6: Train the model on the training data
model.fit(X_train_tfidf, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluate the model
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# In[19]:


# Print classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# In[21]:


# Step 9: Plot the Confusion Matrix
# Visualize the confusion matrix to better understand predictions
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[28]:


# Step 10: Example predictions on new reviews
# Input some example reviews to test the model
example_reviews = [
    "The product quality is excellent!",
    "Horrible experience, will not return.",
    "Decent product but expected better.",
    "Absolutely love it! Fast delivery too."
]

# Transform the example reviews using the TF-IDF vectorizer
example_tfidf = vectorizer.transform(example_reviews)

# Predict the sentiment for each example review
example_preds = model.predict(example_tfidf)

# Display predictions
for review, sentiment in zip(example_reviews, example_preds):
    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    print(f"Review: '{review}' | Sentiment: {sentiment_label}")


# In[33]:


# Step 11: Visualize Example Predictions
# Create a bar plot to visualize the sentiment predictions
sentiments = ["Positive" if s == 1 else "Negative" for s in example_preds]
plt.figure(figsize=(8, 4))
sns.barplot(x=sentiments, y=[1]*len(sentiments), palette="viridis", dodge=False)
plt.title("Sentiment Predictions for Example Reviews")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Count")
plt.show()

