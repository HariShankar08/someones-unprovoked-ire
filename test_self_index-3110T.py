import os
from typing import Iterable, Tuple

# Import the main index class
from self_index import SelfIndex

# --- Configuration ---

# 1. SET YOUR 'xyziq' CONFIGURATION HERE
# This is the base config: (x=1, y=1, z=1, i=0, q=T)
# x=1: BOOLEAN, x=2: WORDCOUNT, x=3: TFIDF
INFO = 'TFIDF'
# y=1: CUSTOM, y=2: DB1, y=3: DB2
DSTORE = 'CUSTOM'
# z=1: NONE, z=2: CODE, z=3: CLIB
COMPR = 'NONE'
# i=0: Null, i=1: Skipping (osp)
OPTIM = 'Null'
# q=T: TERMatat, q=D: DOCatat
QPROC = 'TERMatat'

# 2. SET THE DATA PATH
# Assumes you have a folder named 'data' with .txt files
DATA_DIR = "./data"

# 3. SET THE INDEX ID
# This will be the name of the index file (e.g., test_index.pkl)
INDEX_ID = "test_index"

# 4. SET TEST QUERIES
# These will be run after indexing
TEST_QUERIES = [
    '"Apple"',                          # Simple term
    '"Apple" AND "Banana"',             # AND query
    '"Apple" OR "Orange"',              # OR query
    'NOT "Grape"',                      # NOT query
    '("Apple" AND "Banana") OR "Orange"', # Grouping
    '"quick brown"'                     # Phrase query (only for i1)
]

# --- Helper Function ---

def load_files_from_dir(path: str) -> Iterable[Tuple[str, str]]:
    """
    Loads all .txt files from a directory.
    Yields tuples of (file_id, file_content).
    """
    if not os.path.exists(path):
        print(f"Error: Data directory not found at '{path}'")
        print("Please create it and add some .txt files.")
        return
        
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)
            # Use filename as the file_id
            file_id = filename
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                yield (file_id, content)

# --- Main Test Script ---

def main():
    
    print("--- 1. Initializing Index ---")
    try:
        idx = SelfIndex(
            info=INFO,
            dstore=DSTORE,
            compr=COMPR,
            qproc=QPROC,
            optim=OPTIM
        )
        print(f"Successfully initialized index: {idx}")
    except Exception as e:
        print(f"Failed to initialize index: {e}")
        return

    print("\n--- 2. Loading Data Files ---")
    files_to_index = list(load_files_from_dir(DATA_DIR))
    if not files_to_index:
        print("No files found to index. Exiting.")
        return
    print(f"Loaded {len(files_to_index)} files from '{DATA_DIR}'.")
    
    print("\n--- 3. Creating Index ---")
    try:
        idx.create_index(INDEX_ID, files_to_index)
        print(f"Index '{INDEX_ID}' created successfully.")
    except Exception as e:
        print(f"Failed to create index: {e}")
        return
        
    print(f"Indexed files: {idx.list_indexed_files(INDEX_ID)}")
    
    # --- Clear index from memory to test loading ---
    del idx
    
    print("\n--- 4. Loading Index from Disk ---")
    # Re-initialize the *same configuration*
    idx_loader = SelfIndex(
        info=INFO,
        dstore=DSTORE,
        compr=COMPR,
        qproc=QPROC,
        optim=OPTIM
    )
    
    if idx_loader.load_index(INDEX_ID):
        print(f"Successfully loaded '{INDEX_ID}' from disk.")
    else:
        print(f"Failed to load '{INDEX_ID}' from disk.")
        return

    print("\n--- 5. Running Test Queries ---")
    for q in TEST_QUERIES:
        try:
            results_json = idx_loader.query(q)
            print(f"Query: {q}\nResult: {results_json}\n")
        except Exception as e:
            print(f"Query failed: {q}\nError: {e}\n")

    print("\n--- 6. Cleaning Up ---")
    try:
        idx_loader.delete_index(INDEX_ID)
        print(f"Successfully deleted index '{INDEX_ID}'.")
    except Exception as e:
        print(f"Failed to delete index: {e}")

    print("\n--- Test Run Complete ---")


if __name__ == "__main__":
    # --- Create dummy data if 'data' dir is missing ---
    if not os.path.exists(DATA_DIR):
        print(f"'{DATA_DIR}' not found, creating dummy data...")
        os.makedirs(DATA_DIR, exist_ok=True)
        dummy_data = {
            "doc1.txt": "Apple and Banana are fruits.",
            "doc2.txt": "Apple and Orange are also fruits.",
            "doc3.txt": "Banana is yellow. Orange is orange.",
            "doc4.txt": "I like to eat a quick brown fox.",
            "doc5.txt": "I like to eat a quick brown apple."
        }
        for fname, fcontent in dummy_data.items():
            with open(os.path.join(DATA_DIR, fname), 'w') as f:
                f.write(fcontent)
    
    main()