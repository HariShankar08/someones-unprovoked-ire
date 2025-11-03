from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)
ds = ds.take(1_000)


for example in tqdm(ds, desc=f"Writing files:"):
    title = example['title']
    title = title.replace('/', '_')
    text = example['text']
    with open(f'data/{title}.txt', 'w', encoding='utf-8') as f:
        f.write(text)
