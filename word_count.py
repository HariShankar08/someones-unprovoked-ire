import os
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def download_nltk_resources():
    """
    Downloads necessary NLTK data files if not already present.
    """
    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet'
    }
    for path, resource_id in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"NLTK resource '{resource_id}' not found. Downloading...")
            nltk.download(resource_id, quiet=True)
            print(f"Finished downloading '{resource_id}'.")

def process_files(data_dir='data'):
    """
    Processes all text files in the data directory to count word frequencies
    in two ways:
    1. Original: Simple lowercase and punctuation stripping.
    2. Preprocessed: Tokenized, stopword removal, and lemmatization.
    """

    # --- Setup ---
    # Ensure NLTK resources are available
    download_nltk_resources()

    # Initialize tools and dictionaries
    lemmatizer = WordNetLemmatizer()
    # Use a set for faster lookups
    stop_words = set(stopwords.words('english'))
    # Punctuation from the original script
    punctuation_to_strip = '.,!?;"()[]{}'

    # Dictionaries for both counts
    word_freq_original = {}
    word_freq_preprocessed = {}

    # --- Directory Check ---
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        print("Please create the 'data' directory and add your text files.")
        return

    files = os.listdir(data_dir)
    if not files:
        print(f"Warning: No files found in '{data_dir}'.")
        return

    print(f"Found {len(files)} files. Starting processing...")

    # --- Main File Processing Loop ---
    for filename in tqdm(files):
        filepath = os.path.join(data_dir, filename)

        # Ensure it's a file, not a subdirectory (basic check)
        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

                # === 1. Original Logic ===
                # Use the same .split() logic as the original script
                words_original = text.split()
                for word in words_original:
                    # .strip() only removes from ends
                    word_clean = word.lower().strip(punctuation_to_strip)
                    if word_clean:
                        word_freq_original[word_clean] = word_freq_original.get(word_clean, 0) + 1

                # === 2. New Preprocessing Logic ===
                # Use NLTK's tokenizer for better word separation
                tokens = word_tokenize(text)
                for word in tokens:
                    word_lower = word.lower()

                    # Filter: must be alphabetic (removes punctuation/numbers)
                    # and must not be a stopword.
                    if word_lower.isalpha() and word_lower not in stop_words:
                        # Lemmatize the word
                        lemma = lemmatizer.lemmatize(word_lower)
                        word_freq_preprocessed[lemma] = word_freq_preprocessed.get(lemma, 0) + 1

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # --- Save Results to JSON ---

    # 1. Save original count
    try:
        with open('word_count.json', 'w', encoding='utf-8') as f:
            json.dump(word_freq_original, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved original word count to 'word_count.json'")
    except IOError as e:
        print(f"Error writing to 'word_count.json': {e}")

    # 2. Save new preprocessed count
    try:
        with open('word_count_pre.json', 'w', encoding='utf-8') as f:
            json.dump(word_freq_preprocessed, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved preprocessed word count to 'word_count_pre.json'")
    except IOError as e:
        print(f"Error writing to 'word_count_pre.json': {e}")

# --- Run the script ---
if __name__ == "__main__":
    # Assuming the data is in a folder named 'data'
    # in the same directory as this script.
    process_files('data')