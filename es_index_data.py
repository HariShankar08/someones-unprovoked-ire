from es_index import ElasticSearchIndex
import os
from typing import List, Tuple
from tqdm import tqdm

index_name = 'esindex-v1.0'  # Elasticsearch prefers lowercase index names


def build_file_tuples(data_dir: str) -> List[Tuple[str, str]]:
    """Read up to `limit` text files from `data_dir` and return list of
    (file_id, content) tuples. file_id is the filename without extension.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    files = sorted(files)

    result: List[Tuple[str, str]] = []
    for fname in tqdm(files, desc="Building file tuples"):
        path = os.path.join(data_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                content = fh.read()
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")
            continue

        file_id = os.path.splitext(fname)[0]
        result.append((file_id, content))

    return result


if __name__ == "__main__":
    # Instantiate ElasticSearchIndex with required params.
    # These strings match the expected constructor signature in `es_index.py`.
    idx = ElasticSearchIndex(
        info='WORDCOUNT',
        dstore='DB1',
        qproc='TERMatat',
        compr='NONE',
        optim='Null',
    )

    # Build test documents (file_id, content)
    data_dir = 'data'
    if not os.path.isdir(data_dir):
        print(f"Data directory '{data_dir}' not found. Nothing to index.")
        raise SystemExit(1)

    docs = build_file_tuples(data_dir)
    if not docs:
        print("No documents found to index. Exiting.")
        raise SystemExit(1)

    # 0. If ES connection not available, bail out early with a clear message
    if not idx.es:
        print("Elasticsearch client not available. Check ES server and retry.")
        raise SystemExit(1)

    # 1. Delete any existing test index (ignore errors)
    print(f"Deleting index '{index_name}' (if it exists)...")
    idx.delete_index(index_name)

    # 2. Create index and index our sample documents
    print(f"Creating index '{index_name}' and indexing {len(docs)} documents...")
    idx.create_index(index_id=index_name, files=docs)

    # 3. List indices (user indices only) and show whether our index is present
    print("Listing indices:")
    indices = idx.list_indices()
    print(indices)

    # 4. List files in the created index
    print(f"Listing documents in index '{index_name}':")
    files_in_index = idx.list_indexed_files(index_id=index_name)
    print(files_in_index)

    # 5. Run a sample query against the index
    sample_query = 'alchemy'
    print(f"Querying index '{index_name}' for: {sample_query}")
    results = idx.query(sample_query)
    print(results)

    # 6. Cleanup: delete the test index
    print(f"Cleaning up: deleting index '{index_name}'...")
    idx.delete_index(index_name)

    print('Done.')