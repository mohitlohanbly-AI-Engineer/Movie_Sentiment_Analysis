import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure you've run this once in Python shell:
# import nltk
# nltk.download("stopwords")
# nltk.download("wordnet")

# Load resources
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Clean and preprocess text for sentiment analysis.
    Steps:
      - Lowercase
      - Remove URLs
      - Remove non-alphabetic characters
      - Tokenize
      - Remove stopwords
      - Lemmatize words
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove non-alphabet characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords + lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# Quick test (run only when executing this file directly)
if __name__ == "__main__":
    sample = "NLTK is running!!! Visit https://nltk.org for more info."
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
