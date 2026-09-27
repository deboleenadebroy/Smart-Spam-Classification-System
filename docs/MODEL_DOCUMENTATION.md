# Smart Spam Classification System — Model Documentation

## 1. Project Overview

The Smart Spam Classification System is a machine learning-based system that classifies SMS messages as either **ham (legitimate)** or **spam**.

The system uses text preprocessing, TF-IDF vectorization, and Logistic Regression for classification.

---

## 2. Dataset

The project uses the **SMS Spam Collection dataset**.

Each message belongs to one of two classes:

- **ham** — legitimate/non-spam message
- **spam** — unwanted or fraudulent message

The dataset contains SMS text that is used to train and evaluate the classification model.

---

## 3. Text Preprocessing

Before the messages are given to the machine learning model, they are cleaned.

The preprocessing includes:

1. Converting text to lowercase.
2. Removing URLs.
3. Removing non-alphabetic characters.
4. Removing extra whitespace.
5. Stripping leading and trailing whitespace.

This produces cleaner and more consistent text for feature extraction.

---

## 4. TF-IDF Vectorization

Machine learning models cannot directly process raw text.

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text messages into numerical feature vectors.

The vectorizer is configured with:

- English stop-word removal
- Maximum of 5,000 features

The TF-IDF vectorizer is fitted on the training data and then used to transform both training and test messages.

---

## 5. Logistic Regression Classifier

The classification model used is **Logistic Regression**.

It learns patterns in the TF-IDF features and predicts whether a message belongs to the `ham` or `spam` class.

The model is configured with:

- `max_iter = 1000`
- `random_state = 42`

The trained model is then used for predictions on unseen messages.

---

## 6. Model Evaluation

The model was evaluated using a held-out test set.

The following metrics were obtained:

| Metric | Result |
|---|---:|
| Accuracy | 96.50% |
| Precision | 100.00% |
| Recall | 73.83% |
| F1-Score | 84.94% |

### Accuracy

Accuracy represents the proportion of all test messages that were classified correctly.

### Precision

Precision represents the proportion of messages predicted as spam that were actually spam.

### Recall

Recall represents the proportion of actual spam messages that the model successfully detected.

### F1-Score

F1-score combines precision and recall into a single metric using their harmonic mean.

---

## 7. Prediction Probability

The trained Logistic Regression model can provide prediction probabilities.

For a message, the system can return:

- Predicted class
- Spam probability
- Overall prediction confidence

For example, a test message can produce a spam probability such as `0.9293`, corresponding to approximately **92.93%**.

---

## 8. Saved Model Artifacts

The trained components are saved in the `artifacts/` directory:

```text
artifacts/
├── spam_classifier.joblib
└── tfidf_vectorizer.joblib