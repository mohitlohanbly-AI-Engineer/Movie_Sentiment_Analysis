
# 🎭 Sentiment Analysis System

A **robust sentiment analysis application** designed to classify text reviews from **IMDb** or **Amazon** into **Positive, Negative, or Neutral** categories.
This system combines **traditional ML models (Logistic Regression, SVM)** with **state-of-the-art Transformer models (BERT)** to ensure accurate sentiment predictions.

---

## 📌 Key Features

* **Multiple Models:** Logistic Regression, SVM, and BERT for ensemble-like predictions.
* **Preprocessing Pipeline:** Cleans text, removes noise, and applies TF-IDF vectorization.
* **Interactive Web App:** Built with Streamlit for real-time sentiment analysis.
* **Evaluation Metrics:** Accuracy, classification report, and confusion matrix visualizations.
* **Lightweight & Scalable:** Supports rapid training on subsets or full datasets.

---

## 🗂 Project Structure

```
Sentiment Analysis System/
│── app.py                 # Streamlit interface for user interaction
│── train_models.py        # Script to preprocess data and train ML models
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
│
├── data/
│   └── IMDB.csv           # Dataset containing movie reviews & sentiments
│
├── models/
│   ├── logreg_model.pkl   # Trained Logistic Regression model
│   ├── svm_model.pkl      # Trained SVM model
│   └── vectorizer.pkl     # TF-IDF vectorizer used for training
│
└── utils/
    └── preprocessing.py   # Text cleaning and preprocessing functions
```

---

## 🔄 Workflow

```mermaid
flowchart TD
    A[📂 Dataset: IMDB.csv] --> B[🧹 Text Preprocessing]
    B --> C[🔠 TF-IDF Vectorization]
    C --> D1[🤖 Logistic Regression Training]
    C --> D2[🤖 SVM Training]
    D1 & D2 --> E[💾 Save Models & Vectorizer]
    E --> F[📊 Model Evaluation]Perfect! I’ve reworked the README to be fully **formal, text-only, professional**, without emojis or images, suitable for GitHub or interview presentation.

---

# Sentiment Analysis System

A sentiment analysis application designed to classify text reviews from IMDb or Amazon into Positive, Negative, or Neutral categories. The system combines traditional machine learning models (Logistic Regression, SVM) with a Transformer-based model (BERT) for accurate sentiment prediction.

---

## Key Features

* Multiple Models: Logistic Regression, SVM, and BERT for improved prediction accuracy.
* Text Preprocessing: Cleans and normalizes text, removing noise and irrelevant characters.
* Interactive Web Interface: Streamlit-based real-time sentiment analysis.
* Evaluation Metrics: Accuracy, classification report, and confusion matrix visualization.
* Scalable: Supports full dataset training or smaller subsets for faster experimentation.

---

## Project Structure

```
Sentiment Analysis System/
│── app.py                 # Streamlit interface for real-time sentiment analysis
│── train_models.py        # Script for data preprocessing, training, and saving models
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
│
├── data/
│   └── IMDB.csv           # Dataset containing movie reviews and their sentiment labels
│
├── models/
│   ├── logreg_model.pkl   # Saved Logistic Regression model
│   ├── svm_model.pkl      # Saved SVM model
│   └── vectorizer.pkl     # TF-IDF vectorizer used for training
│
└── utils/
    └── preprocessing.py   # Text cleaning and preprocessing functions
```

---

## Workflow

1. **Load Dataset:** Reviews and sentiments are loaded from `IMDB.csv`.
2. **Preprocess Text:** Convert to lowercase, remove punctuation and special characters, and normalize spacing.
3. **Feature Extraction:** TF-IDF vectorization converts text to numerical features for model training.
4. **Model Training:**

   * Logistic Regression trained on TF-IDF features.
   * Linear SVM trained on the same features.
   * Optional BERT pipeline for semantic understanding of the text.
5. **Model Evaluation:** Accuracy, precision, recall, F1-score, and confusion matrices are generated for both models.
6. **Deployment:** Streamlit web app allows users to input reviews and receive sentiment predictions from all models.

---

## Installation and Setup

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/sentiment-analysis-system.git
cd sentiment-analysis-system
```

2. **Create a virtual environment and activate it:**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Train models (optional if pre-trained models exist):**

```bash
python train_models.py
```

5. **Run the Streamlit app:**

```bash
streamlit run app.py
```

Access the app at `http://localhost:8501`.

---

## Example Usage

* **Input:**
  `"This movie was excellent with outstanding performances and visuals."`

* **Output:**

  * Logistic Regression: Positive
  * SVM: Positive
  * BERT: Positive (Confidence: 99.7%)

---

## Evaluation Metrics

* Logistic Regression Accuracy: Approximately 85%
* SVM Accuracy: Approximately 87%
* Confusion matrices and classification reports generated during training for detailed error analysis

---

## Technology Stack

* Python 3.10+
* scikit-learn: Logistic Regression, SVM, TF-IDF vectorization
* HuggingFace Transformers: BERT for sentiment analysis
* Streamlit: Web application interface
* Seaborn & Matplotlib: Data visualization
* NLTK & Regex: Text preprocessing

---

## Notes

* Pre-trained models are stored in the `models/` folder for quick deployment.
* Designed for scalability: can be extended to other review datasets or multiple languages.
* Provides clean separation between training and deployment workflows.

---

If you want, I can also **draft a corresponding professional “Project Flow” section in text format for the README** that describes step-by-step operations in a concise paragraph form.

Do you want me to do that?

    F --> G[🌐 Streamlit Web App]
    G --> H[✍️ User Review Input]
    H --> I[📌 Sentiment Prediction: LogReg, SVM, BERT]
    I --> J[✅ Display Final Sentiment with Confidence Scores]
```

---

## 🏗 Technical Overview

1. **Data Preprocessing:**

   * Converts text to lowercase
   * Removes punctuation, special characters, and excess whitespace
   * Applies optional lemmatization or stopword removal

2. **Feature Extraction:**

   * Uses **TF-IDF Vectorizer** to convert text into numerical features
   * Limits vocabulary for performance optimization

3. **Model Training:**

   * Logistic Regression & Linear SVM trained on preprocessed TF-IDF features
   * Optional Transformer (BERT) pipeline for semantic understanding

4. **Evaluation:**

   * Accuracy, precision, recall, and F1-score
   * Confusion matrix visualization for error analysis

5. **Deployment:**

   * Streamlit interface allows users to input reviews
   * Displays predictions from all models and BERT confidence score

---

## 🚀 Installation & Setup

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/sentiment-analysis-system.git
cd sentiment-analysis-system
```

2. **Create virtual environment:**

```bash
python -m venv .venv
# Activate
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Train models (optional if pre-trained models exist):**

```bash
python train_models.py
```

5. **Run the Streamlit app:**

```bash
streamlit run app.py
```

Open the browser at **[http://localhost:8501](http://localhost:8501)**

---

## 📊 Example Usage

* **Input:**

  > "This movie was absolutely fantastic with brilliant performances and visuals!"

* **Output:**

  * Logistic Regression → Positive
  * SVM → Positive
  * BERT → Positive (Confidence: 99.7%)

---

## 📈 Evaluation Metrics

* Logistic Regression Accuracy: **\~85%**
* SVM Accuracy: **\~87%**
* Confusion matrices and classification reports generated during training for detailed analysis

---

## 🛠 Tech Stack

* **Python 3.10+**
* **scikit-learn** (Logistic Regression, SVM, TF-IDF)
* **HuggingFace Transformers** (BERT for sentiment analysis)
* **Streamlit** (web app interface)
* **Seaborn & Matplotlib** (visualizations)
* **NLTK & Regex** (text preprocessing)

---

## 🔖 Notes

* Pre-trained models are stored in `models/` for quick deployment.
* Supports batch review analysis by modifying `app.py`.
* Designed for scalability: can be extended to **other review datasets** or **multi-language support**.

