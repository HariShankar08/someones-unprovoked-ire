import json
from typing import Iterable
from elasticsearch import Elasticsearch, NotFoundError

# Import the base class from the other file
from index_base import IndexBase, IndexInfo, DataStore, Compression, QueryProc, Optimizations

# --- NLTK Preprocessing ---
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
# --- End NLTK ---


class ElasticSearchIndex(IndexBase):
    """
    Elasticsearch implementation of the IndexBase.
    
    This class connects to an Elasticsearch instance and implements
    the abstract methods for creating, managing, and querying indices.
    """
    
    def __init__(self, info, dstore, qproc, compr, optim):
        """
        Initializes the Elasticsearch index implementation.
        """
        # Set up the identifier using the base class
        super().__init__(core='ESIndex', info=info, dstore=dstore, qproc=qproc, compr=compr, optim=optim)
        
        # Download NLTK data (stopwords, tokenizer, lemmatizer)
        download_nltk_resources()
        
        # Initialize NLTK tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize Elasticsearch client
        try:
            # Assumes Elasticsearch is running on http://localhost:9200
            # Explicitly pass the host, as required by newer versions of the client
            self.es = Elasticsearch('http://localhost:9200') 
            self.es.ping()
            print("Successfully connected to Elasticsearch.")
        except Exception as e:
            print(f"Could not connect to Elasticsearch: {e}")
            print("Please ensure Elasticsearch is running on localhost:9200.")
            self.es = None # Set to None to handle errors in other methods
            
        # This will hold the name of the currently active index
        self.active_index_id = None

    def _preprocess_text(self, text: str) -> str:
        """
        Internal helper to preprocess text (tokenize, lowercase, stopword, lemmatize).
        """
        tokens = word_tokenize(text)
        lemmas = []
        for word in tokens:
            word_lower = word.lower()
            if word_lower.isalpha() and word_lower not in self.stop_words:
                lemma = self.lemmatizer.lemmatize(word_lower)
                lemmas.append(lemma)
        return " ".join(lemmas)

    def create_index(self, index_id: str, files: Iterable[tuple[str, str]]) -> None:
        if not self.es:
            print("Cannot create index: Elasticsearch connection not available.")
            return

        # Elasticsearch prefers lowercase index names
        index_id = index_id.lower()

        try:
            # Create the index if it doesn't exist
            if not self.es.indices.exists(index=index_id):
                self.es.indices.create(index=index_id)
                print(f"Created index '{index_id}'")
            else:
                print(f"Index '{index_id}' already exists. Indexing files...")
            
            # Index each file
            for file_id, content in tqdm(files, desc="Indexing files"):
                preprocessed_content = self._preprocess_text(content)
                
                doc = {
                    'raw_text': content,
                    'preprocessed_text': preprocessed_content
                }
                
                # Use file_id as the document's _id for easy retrieval/deletion
                self.es.index(index=index_id, id=file_id, document=doc)
            
            # Refresh the index to make documents searchable immediately
            self.es.indices.refresh(index=index_id)
            print(f"Successfully indexed {len(list(files))} documents into '{index_id}'.")
            
            # Set this as the active index
            self.active_index_id = index_id

        except Exception as e:
            print(f"Error creating/indexing: {e}")

    def load_index(self, index_name: str) -> None:
        """
        For Elasticsearch, 'loading' just means setting the active index name
        and verifying it exists.
        """
        if not self.es:
            print("Cannot load index: Elasticsearch connection not available.")
            return
            
        index_name = index_name.lower()
        
        try:
            if self.es.indices.exists(index=index_name):
                self.active_index_id = index_name
                print(f"Set active index to '{index_name}'.")
            else:
                print(f"Warning: Index '{index_name}' does not exist in Elasticsearch.")
                self.active_index_id = None
        except Exception as e:
            print(f"Error checking index: {e}")

    def update_index(self, index_id: str, remove_files: Iterable[tuple[str, str]], add_files: Iterable[tuple[str, str]]) -> None:
        if not self.es:
            print("Cannot update index: Elasticsearch connection not available.")
            return
            
        index_id = index_id.lower()
        
        # --- Remove files ---
        removed_count = 0
        for file_id, _ in remove_files: # Content isn't needed for deletion
            try:
                self.es.delete(index=index_id, id=file_id)
                removed_count += 1
            except NotFoundError:
                print(f"Could not remove doc '{file_id}': Not found.")
            except Exception as e:
                print(f"Error removing doc '{file_id}': {e}")
        if removed_count > 0:
            print(f"Removed {removed_count} documents.")

        # --- Add files ---
        added_count = 0
        for file_id, content in add_files:
            try:
                preprocessed_content = self._preprocess_text(content)
                doc = {
                    'raw_text': content,
                    'preprocessed_text': preprocessed_content
                }
                # .index() will create or update the document
                self.es.index(index=index_id, id=file_id, document=doc)
                added_count += 1
            except Exception as e:
                print(f"Error adding doc '{file_id}': {e}")
        if added_count > 0:
            print(f"Added/Updated {added_count} documents.")

        # Refresh the index
        if removed_count > 0 or added_count > 0:
            self.es.indices.refresh(index=index_id)
            print(f"Index '{index_id}' refreshed.")

    def query(self, query: str) -> str:
        if not self.es:
            return json.dumps({"error": "Elasticsearch connection not available."})
            
        if not self.active_index_id:
            return json.dumps({"error": "No index loaded. Call create_index() or load_index() first."})

        try:
            # Preprocess the query string
            processed_query = self._preprocess_text(query)
            
            # Build an ES 'match' query
            es_query = {
                'query': {
                    'match': {
                        'preprocessed_text': processed_query
                    }
                }
            }
            
            # Execute the search
            response = self.es.search(index=self.active_index_id, body=es_query)
            
            # Format the results
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'source': hit['_source']
                })
                
            return json.dumps(results, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Query failed: {e}"})

    def delete_index(self, index_id: str) -> None:
        if not self.es:
            print("Cannot delete index: Elasticsearch connection not available.")
            return
            
        index_id = index_id.lower()
        
        try:
            self.es.indices.delete(index=index_id, ignore=[400, 404])
            print(f"Successfully deleted index '{index_id}'.")
            
            if self.active_index_id == index_id:
                self.active_index_id = None # Clear active index if it was deleted
        except Exception as e:
            print(f"Error deleting index: {e}")

    def list_indices(self) -> Iterable[str]:
        if not self.es:
            print("Cannot list indices: Elasticsearch connection not available.")
            return []
            
        try:
            # Get indices info
            indices = self.es.cat.indices(format="json")
            
            # Filter out system indices (starting with '.')
            user_indices = [idx['index'] for idx in indices if not idx['index'].startswith('.')]
            return user_indices
        except Exception as e:
            print(f"Error listing indices: {e}")
            return []

    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        """
        Lists all file/document IDs in the given index.
        Note: This uses a 'match_all' query and returns document _ids.
        """
        if not self.es:
            print("Cannot list files: Elasticsearch connection not available.")
            return []
            
        index_id = index_id.lower()
            
        try:
            # Refresh to make sure we get all docs
            self.es.indices.refresh(index=index_id)
        
            # Query for all docs, requesting only the _id (no _source)
            # Setting size to 1000 as a practical limit for this example.
            # For >10k docs, a scroll/search_after query would be needed.
            query_all = {
                'query': {'match_all': {}},
                '_source': False, # Don't return the document content
                'size': 1000
            }
            
            response = self.es.search(index=index_id, body=query_all)
            
            # Extract the _id from each hit
            file_ids = [hit['_id'] for hit in response['hits']['hits']]
            for file in file_ids:
                # Ensure that file "data/{}.txt" is a file that exists.
                file_path = f"data/{file}.txt"
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                    pass
            return [f'data/{file}.txt' for file in file_ids]
            
        except NotFoundError:
            print(f"Index '{index_id}' not found.")
            return []
        except Exception as e:
            print(f"Error listing indexed files: {e}")
            return []


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Initializing ElasticSearchIndex ---")
    # Initialize the index with some sample enum values
    idx = ElasticSearchIndex(
        info='WORDCOUNT', 
        dstore='DB1', 
        qproc='TERMatat', 
        compr='NONE', 
        optim='Null'
    )
    print(idx) # This will print the identifier
    
    # Check if connection was successful
    if idx.es:
        index_name = "my-test-index-v1"
        
        # --- 1. Create Index ---
        print(f"\n--- 1. Creating Index '{index_name}' ---")
        dummy_files = [
            ('doc_a', 'This is the first sample document about Python.'),
            ('doc_b', 'The second document is all about Elasticsearch.'),
            ('doc_c', 'Python and Elasticsearch are a powerful combination.')
        ]
        idx.create_index(index_id=index_name, files=dummy_files)
        
        # --- 2. List Indices ---
        print("\n--- 2. Listing All Indices ---")
        all_indices = idx.list_indices()
        print(f"Found indices: {all_indices}")
        
        # --- 3. List Indexed Files ---
        print(f"\n--- 3. Listing Files in '{index_name}' ---")
        files_in_index = idx.list_indexed_files(index_id=index_name)
        print(f"Found files: {files_in_index}")
        
        # --- 4. Query Index ---
        # Note: create_index already set the active_index_id
        print("\n--- 4. Querying Index for 'powerful python' ---")
        query_results = idx.query("powerful python")
        print(query_results)
        
        # --- 5. Update Index ---
        print("\n--- 5. Updating Index ---")
        files_to_remove = [('doc_a', '')] # Content doesn't matter
        files_to_add = [
            ('doc_b', 'The second document is now about NLTK and Elasticsearch.'), # Update doc_b
            ('doc_d', 'A brand new fourth document.') # Add doc_d
        ]
        idx.update_index(index_id=index_name, remove_files=files_to_remove, add_files=files_to_add)
        
        # --- 6. List Files After Update ---
        print(f"\n--- 6. Listing Files in '{index_name}' (After Update) ---")
        files_in_index_updated = idx.list_indexed_files(index_id=index_name)
        print(f"Found files: {files_in_index_updated}")
        
        # --- 7. Query After Update ---
        print("\n--- 7. Querying Index for 'nltk' (After Update) ---")
        query_results_updated = idx.query("nltk")
        print(query_results_updated)

        # --- 8. Delete Index ---
        print(f"\n--- 8. Deleting Index '{index_name}' ---")
        idx.delete_index(index_id=index_name)
        
        # --- 9. List Indices After Deletion ---
        print("\n--- 9. Listing All Indices (After Deletion) ---")
        final_indices = idx.list_indices()
        print(f"Found indices: {final_indices}")
    
    else:
        print("Skipping example usage due to Elasticsearch connection failure.")

