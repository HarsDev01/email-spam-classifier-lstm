# 📧 Email Spam Classifier using LSTM and NLP

A deep learning-based spam detection system that uses Natural Language Processing (NLP) techniques and a Long Short-Term Memory (LSTM) neural network to classify emails as spam or not spam. This project also includes data visualization, preprocessing, model training, and a prediction interface for user input.

---

## 📌 Features

- 📊 Exploratory Data Analysis with WordClouds and Seaborn plots
- 🔁 Balanced dataset using downsampling for fair training
- 🧹 Text preprocessing (punctuation removal, stopword filtering)
- 🧠 LSTM-based deep learning model with Keras and TensorFlow
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

- The project uses an `emails.csv` dataset with two columns:
  - `text`: Raw email content
  - `spam`: Binary label (0 = Not Spam, 1 = Spam)

---

## 🧪 Model Architecture

- Tokenization and padding of sequences
- `Embedding` Layer
- `LSTM` Layer with 16 units
- Dense Layers with ReLU and Sigmoid activation
- Loss: `BinaryCrossentropy`
- Optimizer: `Adam`
- Early stopping and learning rate reduction used during training

---

## 📈 Performance

- Training and validation accuracy visualized with plots
- Test loss and accuracy printed after evaluation

---

## 🚀 How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/email-spam-classifier.git
   cd email-spam-classifier
