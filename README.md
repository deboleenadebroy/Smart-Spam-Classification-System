# Smart Spam Classification System

A machine-learning based spam message classification system that analyzes text messages and classifies them as **Spam** or **Ham (Legitimate)**.

The project uses **TF-IDF vectorization** and a trained **Logistic Regression classifier**, with a Flask backend that provides real-time message prediction through a web interface/API.

---

## Project Overview

Spam messages are unwanted or potentially deceptive messages that may contain promotional content, fraudulent claims, suspicious links, or other unwanted communication.

The **Smart Spam Classification System** uses Natural Language Processing (NLP) and Machine Learning to automatically analyze the content of a message and determine whether it is:

- **Spam** – potentially unwanted/suspicious message
- **Ham** – legitimate/normal message

The trained machine-learning model is integrated into a Flask application so that users can enter a message and receive a classification result along with its spam probability and confidence.

---

## Features

- Spam and Ham message classification
- Text preprocessing and cleaning
- TF-IDF based text vectorization
- Logistic Regression classification
- Spam probability calculation
- Prediction confidence
- Flask-based backend
- Interactive web interface
- Input validation for empty/invalid messages
- Automated unit testing
- Saved trained model and TF-IDF artifacts for inference

---

## System Workflow

```text
                 User Message
                      |
                      v
              Text Preprocessing
                      |
                      v
             TF-IDF Vectorization
                      |
                      v
            Logistic Regression
                  Classifier
                      |
              +-------+-------+
              |               |
              v               v
            SPAM             HAM
              |               |
              +-------+-------+
                      |
                      v
          Probability + Confidence
                      |
                      v
              Display Result
