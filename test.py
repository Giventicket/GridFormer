"""
import torchtext.datasets as datasets
from pprint import pprint

train, val, test = datasets.Multi30k(language_pair=("de", "en"))
en_text = None
ge_text = None

for idx, ele in enumerate(train):
    if idx == 1:
        break
    ge_text, en_text = ele
"""

from transformers import AutoTokenizer
from pprint import pprint

ge_text = [
    "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.", 
    "Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem."
]

en_text = [
    "Two young, White males are outside near many bushes",
    "Several men in hard hats are operating a giant pulley system."
]

# English tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer(en_text)
print("english")
# pprint(tokens)
# print()

pprint(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
pprint(tokenizer.vocab_size)

# German tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
tokens = tokenizer(ge_text)
print("german")
# pprint(tokens)

pprint(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
pprint(tokenizer.vocab_size)