import json
import os
import math
import pickle
import zlib  # For z=3 (CLIB)
import re    # Imported for robust parsing
import shutil 
import sqlite3 
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable, Dict, List, Any, Set, Tuple

# Attempt to import rocksdbpy, fail gracefully
try:
    import rocksdbpy
    ROCKSDB_AVAILABLE = True
except ImportError:
    ROCKSDB_AVAILABLE = False
    print("Warning: 'rocksdbpy' library not found. d3 (DB2) datastore will not be available.")
    print("Install with: pip install rocksdb-py")

# Assuming index_base.py is in the same directory
from index_base import IndexBase, IndexInfo, DataStore, Compression, QueryProc, Optimizations

# Assuming preprocessing.py with a 'preprocess' function exists
try:
    from preprocessing import preprocess
except ImportError:
    print("Warning: 'preprocessing.py' not found. Using simple whitespace tokenizer.")
    def preprocess(text: str) -> list[str]:
        return text.lower().split()

# --- Globals ---
INDEX_STORAGE_PATH = Path("./index_storage")

# --- Type Aliases for Readability ---
# x=1: {"term": {"doc1": [pos1, pos2], "doc2": [pos3]}}
IndexBoolean = Dict[str, Dict[str, List[int]]]
# x=2: {"term": {"doc1": 4, "doc2": 2}}
IndexWordCount = Dict[str, Dict[str, int]]
# x=3: {"term": {"doc1": 0.45, "doc2": 0.23}}
IndexTFIDF = Dict[str, Dict[str, float]]
# Generic postings dict, value can be list, int, or float
PostingsDict = Dict[str, Any]


class SelfIndex(IndexBase):
    """
    An implementation of IndexBase that uses the 'xyziq' identifier
    to swap implementations for indexing, storage, and querying.
    """

    def __init__(self, info: str, dstore: str, qproc: str, compr: str, optim: str):
        super().__init__(
            core='SelfIndex',
            info=info,
            dstore=dstore,
            qproc=qproc,
            compr=compr,
            optim=optim
        )
        
        # In-memory representation of the index
        self.index: Dict[str, Any] = None
        self.doc_lengths: Dict[str, int] = None
        self.doc_id_skips: Dict[str, Dict[str, str]] = None # For i=1 (osp)
        self.pos_skips: Dict[str, Dict[str, Dict[int, int]]] = None # For i=1 (osp)
        self.current_index_id: str = None
        self.all_doc_ids: Set[str] = set()

        # Check for DB availability
        if "d3" in self.identifier_short and not ROCKSDB_AVAILABLE:
            raise ImportError("RocksDB (d3) selected, but 'rocksdbpy' library is not installed.")

        # Ensure the storage directory exists
        os.makedirs(INDEX_STORAGE_PATH, exist_ok=True)
        # Suppressed print for exhaustive test run
        # print(f"Initialized SelfIndex with config: {self.identifier_short}")

    # --- z=2 (CODE) Helper Methods ---
    
    def _delta_encode(self, int_list: List[int]) -> List[int]:
        """Encodes a sorted list of integers using delta encoding."""
        if not int_list:
            return []
        encoded = [int_list[0]]
        for i in range(1, len(int_list)):
            encoded.append(int_list[i] - int_list[i-1])
        return encoded

    def _delta_decode(self, encoded_list: List[int]) -> List[int]:
        """Decodes a delta-encoded list of integers."""
        if not encoded_list:
            return []
        decoded = [encoded_list[0]]
        for i in range(1, len(encoded_list)):
            decoded.append(decoded[i-1] + encoded_list[i])
        return decoded

    # --- i=1 (osp) Helper Methods ---
    
    def _get_skip_list(self, sorted_list: List[Any]) -> Dict[int, int]:
        """
        Builds a skip list for a sorted list, returning a dict
        mapping {index_from: index_to}.
        """
        skips = {}
        n = len(sorted_list)
        if n < 3: # Don't bother with tiny lists
            return skips
        
        skip_step = int(math.sqrt(n))
        
        for i in range(0, n - skip_step, skip_step):
            skips[i] = i + skip_step
        
        # Ensure last pointer points to the end
        if skips and list(skips.keys())[-1] != n - 1:
            skips[list(skips.keys())[-1]] = n - 1
            
        return skips

    def _build_skip_pointers(self):
        """Builds skip pointers for docIDs and positions (if i1)."""
        print("Building skip pointers...")
        self.doc_id_skips = defaultdict(dict)
        self.pos_skips = defaultdict(dict)

        for term, postings in self.index.items():
            # 1. Build skips for Document IDs
            # We must sort doc IDs to have a defined order
            sorted_doc_ids = sorted(postings.keys())
            doc_id_skips_indices = self._get_skip_list(sorted_doc_ids)
            # Convert index-to-index to value-to-value for easier use
            self.doc_id_skips[term] = {
                sorted_doc_ids[from_idx]: sorted_doc_ids[to_idx]
                for from_idx, to_idx in doc_id_skips_indices.items()
            }

            # 2. Build skips for Positions (only for x=1 / i1)
            if "i1" in self.identifier_short:
                self.pos_skips[term] = {}
                for doc_id, positions in postings.items():
                    # Positions are already sorted by create_index
                    pos_skips_indices = self._get_skip_list(positions)
                    self.pos_skips[term][doc_id] = pos_skips_indices

    # --- Persistence (y=n, z=n) Helper Methods ---

    def _get_storage_path(self, index_id: str) -> Path:
        """
        Gets the storage file/dir path based on datastore choice (y=n).
        """
        # y=1: d1 = CUSTOM (pickle file)
        if "d1" in self.identifier_short:
            return INDEX_STORAGE_PATH / f"{index_id}.pkl"
        # y=2: d2 = DB1 (e.g., SQLite)
        elif "d2" in self.identifier_short:
            return INDEX_STORAGE_PATH / f"{index_id}.sqlite"
        # y=3: d3 = DB2 (e.g., RocksDB)
        elif "d3" in self.identifier_short:
            return INDEX_STORAGE_PATH / f"{index_id}.rocksdb"
        raise ValueError(f"Unknown datastore in identifier: {self.identifier_short}")

    def _persist_index(self, index_id: str) -> None:
        storage_path = self._get_storage_path(index_id)
        
        data_to_save = {
            "index": self.index,
            "doc_lengths": self.doc_lengths,
            "all_doc_ids": self.all_doc_ids,
            "doc_id_skips": self.doc_id_skips,
            "pos_skips": self.pos_skips
        }
        
        # print(f"Persisting index '{index_id}' to {storage_path}...")

        # --- z=2 (c2): Simple Compression (CODE) ---
        if "c2" in self.identifier_short:
            # Only compress positional lists (i1)
            if "i1" in self.identifier_short:
                # print("Applying simple compression (Delta Encoding)...")
                for term, postings in self.index.items():
                    for doc_id, positions in postings.items():
                        self.index[term][doc_id] = self._delta_encode(positions)
            # else:
                # print("Skipping simple compression (c2): Not applicable for i2/i3.")
        
        # Serialize data (pickle is the common format before compression/DB)
        serialized_data = pickle.dumps(data_to_save)

        # --- z=3 (c3): Library Compression (CLIB) ---
        if "c3" in self.identifier_short:
            # print("Applying library compression (CLIB/zlib)...")
            serialized_data = zlib.compress(serialized_data)
        
        # z=1 (c1): NONE (Do nothing)

        # --- y=n: Datastore Logic ---

        # y=1 (d1): CUSTOM (pickle)
        if "d1" in self.identifier_short:
            with open(storage_path, "wb") as f:
                f.write(serialized_data)
        
        # y=2 (d2): DB1 (SQLite)
        elif "d2" in self.identifier_short:
            # print("Writing to DB1 (SQLite)...")
            conn = sqlite3.connect(storage_path)
            conn.execute("CREATE TABLE IF NOT EXISTS idx (key TEXT PRIMARY KEY, data BLOB)")
            conn.execute("INSERT OR REPLACE INTO idx (key, data) VALUES (?, ?)", ("main", serialized_data))
            conn.commit()
            conn.close()
            
        # y=3 (d3): DB2 (RocksDB)
        elif "d3" in self.identifier_short:
            # print("Writing to DB2 (RocksDB)...")
            db = None
            try:
                db = rocksdbpy.open(str(storage_path), create_if_missing=True)
                db.put(b'main', serialized_data)
            except Exception as e:
                print(f"Error writing to RocksDB: {e}")
            finally:
                if db:
                    db.close()
            
        # print(f"Successfully persisted '{index_id}'.")

    def load_index(self, index_id: str) -> bool:
        storage_path = self._get_storage_path(index_id)
        if not os.path.exists(storage_path):
            print(f"Error: Index file/dir not found at {storage_path}")
            return False
            
        # print(f"Loading index '{index_id}' from {storage_path}...")
        serialized_data = None

        # --- y=n: Datastore Logic (Read) ---
        
        # y=1 (d1): CUSTOM (pickle)
        if "d1" in self.identifier_short:
            with open(storage_path, "rb") as f:
                serialized_data = f.read()
        
        # y=2 (d2): DB1 (SQLite)
        elif "d2" in self.identifier_short:
            # print("Reading from DB1 (SQLite)...")
            conn = sqlite3.connect(storage_path)
            try:
                cursor = conn.execute("SELECT data FROM idx WHERE key = 'main'")
                row = cursor.fetchone()
                if row:
                    serialized_data = row[0]
            except sqlite3.OperationalError as e:
                print(f"Error reading from SQLite DB: {e}")
            conn.close()
            
        # y=3 (d3): DB2 (RocksDB)
        elif "d3" in self.identifier_short:
            # print("Reading from DB2 (RocksDB)...")
            db = None
            try:
                db = rocksdbpy.open(str(storage_path))
                serialized_data = db.get(b'main')
            except Exception as e:
                print(f"Error reading from RocksDB: {e}")
            finally:
                if db:
                    db.close()
            
        if serialized_data is None:
            print(f"Error: Could not load data for index '{index_id}'")
            return False

        # --- z=n: Decompression Logic ---
        
        # z=3 (c3): CLIB (Library Decompression)
        if "c3" in self.identifier_short:
            # print("Decompressing with CLIB (zlib)...")
            serialized_data = zlib.decompress(serialized_data)
            
        # z=1 (c1): NONE (Do nothing)
        
        # Deserialize
        try:
            data = pickle.loads(serialized_data)
        except Exception as e:
            print(f"Error: Could not deserialize index data. File may be corrupt or compressed differently. {e}")
            return False
        
        self.index = data["index"]
        self.doc_lengths = data["doc_lengths"]
        self.all_doc_ids = data.get("all_doc_ids", set(self.doc_lengths.keys())) # Backwards compatibility
        self.doc_id_skips = data.get("doc_id_skips", defaultdict(dict))
        self.pos_skips = data.get("pos_skips", defaultdict(dict))

        # --- z=2 (c2): Simple Decompression (CODE) ---
        if "c2" in self.identifier_short:
            if "i1" in self.identifier_short:
                # print("Applying simple decompression (Delta Decoding)...")
                for term, postings in self.index.items():
                    for doc_id, encoded_positions in postings.items():
                        self.index[term][doc_id] = self._delta_decode(encoded_positions)
            # else:
                # print("Skipping simple decompression (c2): Not applicable for i2/i3.")
            
        self.current_index_id = index_id
        # print(f"Successfully loaded index '{index_id}'.")
        return True

    # --- Abstract Method Implementations ---

    def create_index(self, index_id: str, files: Iterable[tuple[str, str]]) -> None:
        # print(f"Creating index '{index_id}' with config {self.identifier_short}...")
        all_files = list(files) # Need to iterate multiple times for x=3
        self.all_doc_ids = {file_id for file_id, _ in all_files}
        
        # --- x=n: Index Information Type Logic ---
        
        # x=1: i1 = BOOLEAN (Store doc_id: [positions])
        if "i1" in self.identifier_short:
            self.index: IndexBoolean = defaultdict(dict)
            self.doc_lengths = {}
            from tqdm import tqdm
            for file_id, content in tqdm(all_files):
                terms = preprocess(content)
                self.doc_lengths[file_id] = len(terms)
                for position, term in enumerate(terms):
                    if file_id not in self.index[term]:
                        self.index[term][file_id] = []
                    self.index[term][file_id].append(position)
        
        # x=2: i2 = WORDCOUNT (Store doc_id: term_frequency)
        elif "i2" in self.identifier_short:
            self.index: IndexWordCount = defaultdict(dict)
            self.doc_lengths = {}
            from tqdm import tqdm
            for file_id, content in tqdm(all_files):
                terms = preprocess(content)
                self.doc_lengths[file_id] = len(terms)
                term_counts = Counter(terms)
                for term, count in term_counts.items():
                    self.index[term][file_id] = count
                    
        # x=3: i3 = TFIDF (Store doc_id: tf-idf_score)
        elif "i3" in self.identifier_short:
            self.index: IndexTFIDF = defaultdict(dict) # Stores TF first
            self.doc_lengths = {}
            num_docs = len(all_files)
            
            # Pass 1: Calculate Term Frequencies (TF) and Doc Frequencies (DF)
            df_counts = Counter()
            from tqdm import tqdm
            for file_id, content in tqdm(all_files):
                terms = preprocess(content)
                self.doc_lengths[file_id] = len(terms)
                term_counts = Counter(terms)
                # Use set of terms for doc frequency
                df_counts.update(set(terms)) 
                for term, count in term_counts.items():
                    # Store TF (raw count) for now
                    self.index[term][file_id] = count 
            
            # Pass 2: Calculate TF-IDF
            # print("Calculating TF-IDF scores...")
            for term, postings in tqdm(self.index.items()):
                df = df_counts[term]
                # Standard TF-IDF: log(N / df)
                idf = math.log(num_docs / (1 + df)) + 1 # Use 1+df for smoothing, +1 to avoid 0
                
                for doc_id, tf in postings.items():
                    # TF-IDF calculation
                    # Using augmented TF: 0.5 + (0.5 * tf / max_tf_in_doc)
                    # For simplicity, we'll use log-normalized TF: (1 + log(tf))
                    normalized_tf = (1 + math.log(tf)) if tf > 0 else 0
                    # Store final TF-IDF score
                    self.index[term][doc_id] = normalized_tf * idf
        
        else:
            raise ValueError(f"Unknown index info in identifier: {self.identifier_short}")

        # --- i=1 (osp): Index Optimization Logic ---
        if "osp" in self.identifier_short:
            self._build_skip_pointers()

        # Persist the newly created index
        self._persist_index(index_id)
        self.current_index_id = index_id

    def delete_index(self, index_id: str) -> None:
        storage_path = self._get_storage_path(index_id)
        if not os.path.exists(storage_path):
            print(f"Warning: Index file/dir not found at {storage_path}. Cannot delete.")
            return

        # --- y=n: Datastore Logic (Delete) ---
        
        # y=1 (d1) or y=2 (d2)
        if "d1" in self.identifier_short or "d2" in self.identifier_short:
            os.remove(storage_path)
        
        # y=3 (d3)
        elif "d3" in self.identifier_short:
            # This works for both rocksdb and rocksdbpy
            shutil.rmtree(storage_path)
            
        print(f"Deleted index '{index_id}' from {storage_path}.")
        
        # Clear from memory if it was loaded
        if self.current_index_id == index_id:
            
            # --- THIS IS THE BUG FIX ---
            # The old method `int(self.identifier_short[10])` was brittle,
            # failed on non-int or multi-char values (like 'sp' or 'Null').
            # We must re-parse the string robustly.
            
            # Use regex to extract all xyziq parts
            # Handles 'Null' (o0) and 'Skipping' (osp)
            parser_regex = r'.*_i(\d+)d(\d+)c(\d+)q(\w+)o(.*)'
            match = re.match(parser_regex, self.identifier_short)
            
            if not match:
                print("Critical Error: Could not re-parse self.identifier_short. Index state not reset.")
                # Clear memory manually
                self.index = None
                self.doc_lengths = None
                self.current_index_id = None
                return

            i_val, d_val, c_val, q_val, o_val = match.groups()

            # Re-call __init__ using the *names* from the Enums.
            # This is robust and works for '0', 'sp', 'Null', etc.
            
            # Find the Enum key (e.g., 'Null') from its value (e.g., '0')
            optim_name = next((name for name, member in Optimizations.__members__.items() if member.value == o_val), None)
            if optim_name is None:
                # Fallback for integer values like '0'
                 optim_name = next((name for name, member in Optimizations.__members__.items() if member.value == int(o_val)), 'Null')
                 
            self.__init__(
                info=IndexInfo(int(i_val)).name,
                dstore=DataStore(int(d_val)).name,
                compr=Compression(int(c_val)).name,
                qproc=QueryProc(q_val).name,
                optim=optim_name
            )

    def list_indices(self) -> Iterable[str]:
        if not os.path.exists(INDEX_STORAGE_PATH):
            return []
            
        indices = []
        if "d1" in self.identifier_short:
            indices = [f.stem for f in INDEX_STORAGE_PATH.glob("*.pkl")]
        elif "d2" in self.identifier_short:
            indices = [f.stem for f in INDEX_STORAGE_PATH.glob("*.sqlite")]
        elif "d3" in self.identifier_short:
            indices = [f.name for f in INDEX_STORAGE_PATH.glob("*.rocksdb")]
        return indices

    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        if self.current_index_id != index_id:
            if not self.load_index(index_id):
                # print(f"Could not load index '{index_id}' to list files.")
                return []
        return list(self.all_doc_ids)

    def update_index(self, index_id: str, remove_files: Iterable[tuple[str, str]], add_files: Iterable[tuple[str, str]]) -> None:
        print("update_index is not implemented. Please use create_index to rebuild.")
        pass
    
    # ---
    # --- QUERY ENGINE (The complex part)
    # ---

    def query(self, query: str) -> str:
        """
        Queries the *currently loaded* index.
        Swaps engine based on (q=n).
        """
        if self.index is None:
            return json.dumps({"error": "No index is loaded. Call load_index(index_id) first."})
        
        results_data = []

        try:
            # --- q=n: Query Processing Engine Logic ---
            
            # q=D: DOCatat (Document-at-a-Time)
            if "qD" in self.identifier_short:
                # DAAT engine is simpler, assumes AND-only query
                terms = self._parse_query_daat_stub(query)
                results_data = self._execute_daat(terms)
            
            # q=T: TERMatat (Term-at-a-Time)
            elif "qT" in self.identifier_short:
                # TAAT engine uses the full boolean RPN parser
                tokens = self._tokenize_query(query)
                rpn_queue = self._shunting_yard(tokens)
                results_data = self._execute_taat_rpn(rpn_queue)

        except Exception as e:
            return json.dumps({"error": f"Query processing failed: {e}", "query": query})

        # --- Format results ---
        final_results = []
        # x=1 (i1): Boolean, results_data is just a list of doc_ids
        if "i1" in self.identifier_short:
            final_results = results_data
        
        # x=2 (i2) or x=3 (i3): Ranked
        elif "i2" in self.identifier_short or "i3" in self.identifier_short:
            # results_data is a list of (doc_id, score) tuples
            final_results = [doc_id for doc_id, score in results_data]
            
        return json.dumps({"query": query, "results": final_results, "count": len(final_results)})

    # --- TAAT (qT) Engine Methods ---

    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenizes the query string for the shunting-yard parser."""
        # This regex captures:
        # 1. Quoted phrases ("word1 word2")
        # 2. Standalone operators (AND, OR, NOT)
        # 3. Parentheses
        # 4. Unquoted words (which we'll treat as quoted terms)
        token_regex = r'("[^"]+"|\bAND\b|\bOR\b|\bNOT\b|\(|\)|\w+)'
        tokens = re.findall(token_regex, query)
        
        # Wrap unquoted words in quotes to treat them as terms
        processed_tokens = []
        for t in tokens:
            if t not in {"AND", "OR", "NOT", "(", ")"}:
                if not t.startswith('"'):
                    processed_tokens.append(f'"{t}"')
                else:
                    processed_tokens.append(t)
            else:
                processed_tokens.append(t)
        return processed_tokens

    def _shunting_yard(self, tokens: List[str]) -> List[str]:
        """Converts infix query tokens to Reverse Polish Notation (RPN)."""
        output_queue = []
        operator_stack = []
        precedence = {'OR': 1, 'AND': 2, 'NOT': 3}
        
        for token in tokens:
            if token.startswith('"'):  # Operand (term or phrase)
                output_queue.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if not operator_stack:
                    raise ValueError("Mismatched parentheses")
                operator_stack.pop()  # Discard '('
            else:  # Operator (AND, OR, NOT)
                while (operator_stack and 
                       operator_stack[-1] != '(' and 
                       precedence.get(operator_stack[-1], 0) >= precedence[token]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
                
        while operator_stack:
            if operator_stack[-1] == '(':
                raise ValueError("Mismatched parentheses")
            output_queue.append(operator_stack.pop())
            
        return output_queue

    def _execute_taat_rpn(self, rpn_queue: List[str]) -> List[Any]:
        """Executes a query in RPN using Term-at-a-Time logic."""
        eval_stack = []
        
        for token in rpn_queue:
            if token.startswith('"'):
                # Get postings for this term/phrase
                postings = self._get_postings_taat(token)
                eval_stack.append(postings)
            
            elif token == "NOT":
                if not eval_stack:
                    raise ValueError("Invalid NOT operation")
                postings = eval_stack.pop()
                result = self._handle_not(postings)
                eval_stack.append(result)
            
            elif token in ("AND", "OR"):
                if len(eval_stack) < 2:
                    raise ValueError(f"Invalid {token} operation")
                p2 = eval_stack.pop()
                p1 = eval_stack.pop()
                
                if token == "AND":
                    result = self._handle_and(p1, p2)
                else: # OR
                    result = self._handle_or(p1, p2)
                eval_stack.append(result)
        
        if len(eval_stack) != 1:
            raise ValueError("Invalid query structure")
            
        final_postings = eval_stack.pop()

        # Format final result
        if "i1" in self.identifier_short:
            return sorted(list(final_postings.keys()))
        else: # i2, i3
            # Sort by score (value) descending
            return sorted(final_postings.items(), key=lambda item: item[1], reverse=True)

    def _get_postings_taat(self, term_token: str) -> PostingsDict:
        """Gets postings for a single term or phrase token."""
        term_string = term_token.strip('"')
        preprocessed_terms = preprocess(term_string)
        
        if not preprocessed_terms:
            return {}
            
        # Handle Phrase Query
        if len(preprocessed_terms) > 1:
            if "i1" not in self.identifier_short:
                # print("Warning: Phrase queries only supported for i1 (Boolean). Returning empty.")
                return {}
            return self._handle_phrase(preprocessed_terms)
        
        # Handle Single Term
        term = preprocessed_terms[0]
        return self.index.get(term, {})

    def _handle_phrase(self, terms: List[str]) -> PostingsDict:
        """
        Handles a phrase query, e.g., ['quick', 'brown'].
        Only works for i1 (Boolean/Positional).
        """
        if not terms:
            return {}
            
        # Get postings for the first term
        current_postings = self.index.get(terms[0], {})
        result_postings = {}
        
        for doc_id, positions in current_postings.items():
            valid_positions = []
            
            # For each position of term 1...
            for pos in positions:
                # ...check if term 2, 3, etc. follow
                match = True
                for i in range(1, len(terms)):
                    next_term = terms[i]
                    next_term_postings = self.index.get(next_term, {})
                    
                    if doc_id not in next_term_postings:
                        match = False
                        break
                    
                    # Check for position (pos + i)
                    # This is slow (linear scan).
                    # A binary search would be better.
                    if (pos + i) not in next_term_postings[doc_id]:
                        match = False
                        break
                        
                if match:
                    valid_positions.append(pos)
            
            if valid_positions:
                result_postings[doc_id] = valid_positions
                
        return result_postings

    def _handle_and(self, p1: PostingsDict, p2: PostingsDict) -> PostingsDict:
        """AND operation for TAAT."""
        result = {}
        
        # Use skip pointers (osp) if available and lists are long
        if "osp" in self.identifier_short and len(p1) > 10 and len(p2) > 10:
            docs1 = sorted(p1.keys())
            docs2 = sorted(p2.keys())
            
            # Get skips for the *terms* (this is a simplification)
            # A real system would get skips based on the *query terms*
            # This is a stub showing where it *would* go
            # intersection = self._intersect_with_skips(docs1, skips1, docs2, skips2)
            
            # Fallback to standard intersection
            intersection = set(docs1) & set(docs2)
        else:
            # Standard set intersection
            intersection = set(p1.keys()) & set(p2.keys())
        
        # Combine postings/scores
        for doc_id in intersection:
            if "i1" in self.identifier_short:
                result[doc_id] = p1[doc_id] # Just carry over positions
            else: # i2, i3
                result[doc_id] = p1[doc_id] + p2[doc_id] # Add scores
        return result
        
    def _handle_or(self, p1: PostingsDict, p2: PostingsDict) -> PostingsDict:
        """OR operation for TAAT."""
        # Union of keys
        all_docs = set(p1.keys()) | set(p2.keys())
        result = {}
        
        for doc_id in all_docs:
            if "i1" in self.identifier_short:
                # Combine position lists (though rarely useful)
                result[doc_id] = sorted(list(set(p1.get(doc_id, [])) | set(p2.get(doc_id, []))))
            else: # i2, i3
                result[doc_id] = p1.get(doc_id, 0) + p2.get(doc_id, 0) # Add scores
        return result

    def _handle_not(self, p1: PostingsDict) -> PostingsDict:
        """NOT operation for TAAT."""
        docs_to_exclude = set(p1.keys())
        result_docs = self.all_doc_ids - docs_to_exclude
        
        result = {}
        for doc_id in result_docs:
            if "i1" in self.identifier_short:
                result[doc_id] = [] # No positions
            else: # i2, i3
                result[doc_id] = 1.0 # Give a default score
        return result

    # --- DAAT (qD) Engine Methods ---

    def _parse_query_daat_stub(self, query: str) -> List[str]:
        """
        Simple parser for DAAT. Assumes AND-only query.
        e.g., '"term1" AND "term2"' -> ['term1', 'term2']
        """
        tokens = re.findall(r'"([^"]+)"', query)
        preprocessed_terms = []
        for t in tokens:
            preprocessed = preprocess(t)
            if preprocessed:
                preprocessed_terms.append(preprocessed[0])
        return preprocessed_terms

    def _execute_daat(self, terms: List[str]) -> List[Any]:
        """
        Executes an AND query using Document-at-a-Time logic.
        """
        if not terms:
            return []
            
        # Get postings lists for all terms
        postings_lists: List[PostingsDict] = [self.index.get(t, {}) for t in terms]
        
        # Sort lists by size (doc frequency) - smallest first
        postings_lists.sort(key=len)
        
        # Pointers (indices) for each list
        # We need the sorted doc_id lists
        sorted_doc_id_lists = [sorted(p.keys()) for p in postings_lists]
        
        # Skips (if osp)
        doc_id_skips = []
        if "osp" in self.identifier_short:
            doc_id_skips = [self.doc_id_skips.get(t, {}) for t in terms]
            doc_id_skips.sort(key=len) # (This is wrong, need to sort by p-list len)
            # Re-get skips based on the sorted term order
            sorted_terms = sorted(terms, key=lambda t: len(self.index.get(t, {})))
            doc_id_skips = [self.doc_id_skips.get(t, {}) for t in sorted_terms]

        pointers = [0] * len(sorted_doc_id_lists)
        num_lists = len(sorted_doc_id_lists)
        results = {} # {doc_id: score} or {doc_id: []}

        if not sorted_doc_id_lists[0]: # Smallest list is empty
            return []

        while True:
            # Get doc_id from the *first* (smallest) list
            candidate_doc_id = sorted_doc_id_lists[0][pointers[0]]
            
            # Check if this doc_id is in all other lists
            match = True
            next_max_doc_id = candidate_doc_id # Track max doc_id found
            
            for i in range(1, num_lists):
                current_list = sorted_doc_id_lists[i]
                current_ptr = pointers[i]

                # Advance pointer for list 'i' until it's >= candidate_doc_id
                while current_ptr < len(current_list) and current_list[current_ptr] < candidate_doc_id:
                    
                    # --- Use Skip Pointers (osp) ---
                    if "osp" in self.identifier_short:
                        current_doc = current_list[current_ptr]
                        skips_for_this_term = doc_id_skips[i]
                        
                        if current_doc in skips_for_this_term:
                            skip_to_doc = skips_for_this_term[current_doc]
                            if skip_to_doc < candidate_doc_id:
                                # Find new pointer index (slow)
                                # A real skip list would store indices
                                try:
                                    current_ptr = current_list.index(skip_to_doc)
                                except ValueError:
                                    current_ptr += 1 # Failsafe
                                continue # Skip the increment
                    
                    current_ptr += 1

                pointers[i] = current_ptr
                
                # Check for match or end of list
                if current_ptr >= len(current_list):
                    # Reached end of a list, no more matches possible
                    return self._format_daat_results(results)
                    
                if current_list[current_ptr] > candidate_doc_id:
                    # Mismatch. The candidate is not in this list.
                    match = False
                    next_max_doc_id = current_list[current_ptr]
                    break # Stop checking other lists
            
            # --- Process Match / Mismatch ---
            if match:
                # All lists contained candidate_doc_id
                if "i1" in self.identifier_short:
                    results[candidate_doc_id] = [] # Boolean
                else: # i2, i3
                    # Sum scores from all lists
                    score = sum(postings_lists[i][candidate_doc_id] for i in range(num_lists))
                    results[candidate_doc_id] = score
                
                # Advance pointer of the *first* list
                pointers[0] += 1
            else:
                # Mismatch. Jump pointer[0] to at least next_max_doc_id
                current_list_0 = sorted_doc_id_lists[0]
                current_ptr_0 = pointers[0]
                
                while current_ptr_0 < len(current_list_0) and current_list_0[current_ptr_0] < next_max_doc_id:
                    current_ptr_0 += 1
                
                pointers[0] = current_ptr_0
            
            # Check if first list is exhausted
            if pointers[0] >= len(sorted_doc_id_lists[0]):
                break # We're done
        
        return self._format_daat_results(results)

    def _format_daat_results(self, results_dict: Dict) -> List:
        """Helper to format DAAT results."""
        if "i1" in self.identifier_short:
            return sorted(list(results_dict.keys()))
        else:
            return sorted(results_dict.items(), key=lambda item: item[1], reverse=True)