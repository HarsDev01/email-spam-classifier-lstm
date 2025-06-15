# 📧 Email Spam Classifier using LSTM and NLP

A deep learning-based spam detection system that uses Natural Language Processing (NLP) techniques and a Long Short-Term Memory (LSTM) neural network to classify emails as spam or not spam. This project includes data visualization, preprocessing, model training, and a prediction interface for real-time email classification.

---

## 📌 Features

- 📊 Exploratory Data Analysis with WordClouds and Seaborn plots
- 🔁 Balanced dataset using downsampling for fair training
- 🧹 Text preprocessing (punctuation removal, stopword filtering)
- 🧠 LSTM-based deep learning model using Keras and TensorFlow
- ✅ Real-time email spam prediction from user input
- 💾 Model and tokenizer saved for future predictions

---

## 🧰 Libraries Used

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `nltk`, `wordcloud`
- `scikit-learn`
- `tensorflow`, `keras`
- `pickle`

---

## 📂 Dataset

The project uses an `emails.csv` dataset with two columns:
- `text`: Raw email content
- `spam`: Binary label (0 = Not Spam, 1 = Spam)

---

## 🧪 Model Architecture

- Text tokenization and padding
- `Embedding` Layer to learn word representations
- `LSTM` Layer with 16 units to capture sequence dependencies
- Dense layers with ReLU and Sigmoid activations
- Loss: `BinaryCrossentropy`
- Optimizer: `Adam`
- EarlyStopping and ReduceLROnPlateau callbacks for better training control

---

## 📈 Performance

- Training and validation accuracy visualized across epochs
- Final test loss and accuracy printed after evaluation
- Real-time predictions based on new email input from the user

---

## 💾 Output Artifacts

- Trained model saved as: `spam_detector_model.h5`
- Tokenizer saved as: `tokenizer.pickle`
- You can reuse these for deployment or further predictions.

---

## 🧠 Sample Prediction

```plaintext
Enter the email text to check if it is spam: 
Congratulations! You've won a free iPhone! Click here to claim now.

The email is classified as: Spam
