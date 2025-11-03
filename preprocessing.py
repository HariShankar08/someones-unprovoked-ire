import string
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List

# --- NLTK Resource Downloader ---

def download_nltk_resources():
    """
    Checks for and downloads required NLTK data if missing.
    """
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    
    for path, pkg_id in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"NLTK data '{pkg_id}' not found. Downloading...")
            nltk.download(pkg_id)

# Call the downloader on module import to ensure data is ready
download_nltk_resources()

# --- Globals (initialized once for efficiency) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Helper Function for POS Tagging ---

def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Maps NLTK's part-of-speech (POS) tags to the tags expected 
    by the WordNetLemmatizer (NOUN, VERB, ADJ, ADV).
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Default to NOUN if the tag is not recognized
        return wordnet.NOUN

# --- Main Preprocessing Function ---

def preprocess(text: str) -> List[str]:
    """
    Cleans, tokenizes, removes stopwords, and lemmatizes text.
    
    1.  Lowercase
    2.  Remove punctuation and numbers
    3.  Tokenize
    4.  Remove stop words
    5.  Lemmatize based on Part-of-Speech
    
    Args:
        text: The raw input string.
    Returns:
        A list of processed (lemmatized) tokens.
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and numbers
    # Replace with a space to avoid merging words (e.g., "end.Begin")
    text = re.sub(r'[\d' + re.escape(string.punctuation) + ']', ' ', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Get POS tags for lemmatization
    # This is crucial: lemmatizing "running" (Verb) -> "run"
    # vs. lemmatizing "running" (Noun) -> "running"
    pos_tagged_tokens = nltk.pos_tag(tokens)
    
    processed_tokens = []
    for token, tag in pos_tagged_tokens:
        # 5. Remove stop words and short, meaningless tokens
        if token not in stop_words and len(token) > 1:
            
            # 6. Lemmatize with the correct POS tag
            wordnet_pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
            processed_tokens.append(lemma)
            
    return processed_tokens

# --- Self-Test Block ---
if __name__ == "__main__":
    """
    Run this file directly (python preprocessing.py) to test it.
    """
    print("--- Testing NLTK Preprocessing Script ---")
    
    test_text = "The 5 quick brown foxes are jumping over the lazy dogs. They were running and playing."
    
    print(f"\nOriginal:\n{test_text}")
    
    processed_list = preprocess(test_text)
    
    print(f"\nProcessed:\n{processed_list}")
    
    print("\nExpected:")
    print("['quick', 'brown', 'fox', 'jump', 'lazy', 'dog', 'run', 'play']")