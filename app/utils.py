import spacy
import re
import pandas as pd
from bs4 import BeautifulSoup
import emoji  # Make sure to install the emoji library
import string  # Import string for punctuation removal
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer
stemmer = PorterStemmer()

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")
nlp.disable_pipe("parser")
nlp.add_pipe("sentencizer")

# Contractions dictionary
contractions_dict = {
    "ain't": "is not",
    "aren't": "are not",
    # (add the rest of the contractions)
}

# Function to expand contractions
def expand_contractions(text):
    pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group(0)], text)

# Function to remove named entities
def custom_remove_entities(text):
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.ent_type_]
    text_without_spacy_entities = ' '.join(filtered_tokens)
    text_cleaned = re.sub(r'\b[A-Z]\.[A-Za-z]+\b', '', text_without_spacy_entities)  # Regex to remove initials
    return text_cleaned

# Function to remove HTML tags
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")
    return re.sub(r'\s+', ' ', clean_text.strip())  # Clean extra spaces

# Function to remove emojis
def remove_emojis(text):
    clean_text = emoji.replace_emoji(text, replace='')
    return re.sub(r'\s+', ' ', clean_text.strip())  # Clean extra spaces

# Function to remove punctuation
def remove_punctuation(text):
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', clean_text.strip())  # Clean extra spaces

# Function to remove extra whitespace
def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space

# Function to remove numbers (digits and words)
def remove_numbers(text):
    number_words = {
        'zero': '', 'one': '', # (continue the list)
    }

    pattern = r'\b(' + '|'.join(number_words.keys()) + r')\b'
    clean_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\d+', '', clean_text)
    return re.sub(r'\s+', ' ', clean_text.strip())

def handle_negations(text):
    negations = {
        "not good": "bad",
        # (continue the list)
    }
    
    pattern = re.compile(r'\b(' + '|'.join(negations.keys()) + r')\b')
    return pattern.sub(lambda x: negations[x.group(0)], text)

# Main function to process the text
def preprocess_text(text):
    text = expand_contractions(text)
    text = remove_html_tags(text)
    text = remove_emojis(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)
    text = remove_numbers(text)
    text = handle_negations(text)
    text = custom_remove_entities(text)
    return text

# Example usage (you may want to remove this from production)
if __name__ == "__main__":
    sample_text = "I'm not happy with this product! It costs $100."
    processed_text = preprocess_text(sample_text)
    print(processed_text)
