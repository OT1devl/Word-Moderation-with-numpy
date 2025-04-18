import numpy as np
import json
import itertools
import pickle

from settings import tokenizer_path

class CharTokenizer:
    def __init__(self, lower=True, unk_token='<unk>', pad_token='<pad>'):
        self.lower = lower
        self.unk_token = unk_token
        self.pad_token = pad_token
        
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_built = False

    def build_vocab(self, texts):
        chars = set()
        for text in texts:
            if self.lower:
                text = text.lower()
            chars.update(text)

        chars = sorted(list(chars))
        chars = [self.pad_token, self.unk_token] + chars

        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_built = True

    def encode(self, text, max_len=None):
        if self.lower:
            text = text.lower()

        if not self.vocab_built:
            raise ValueError("First you have to build the vocabulary")

        encoded = [self.char2idx.get(ch, self.char2idx[self.unk_token]) for ch in text]

        if max_len:
            if len(encoded) < max_len:
                encoded += [self.char2idx[self.pad_token]] * (max_len - len(encoded))
            else:
                encoded = encoded[:max_len]

        return encoded

    def decode(self, indices, remove_pad=True):
        text = ''.join([self.idx2char.get(idx, self.unk_token) for idx in indices])
        if remove_pad:
            text = text.replace(self.pad_token, '')
        return text

    def vocab_size(self):
        return len(self.char2idx)
    
    def save(self, path: str=tokenizer_path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str=tokenizer_path):
        with open(path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer

def generate_leetspeak_variants(word, num_variants=3):
    replacements = {
        'a': ['4', '@', 'á'],
        'e': ['3', '€', 'é'],
        'i': ['1', '¡', 'í'],
        'o': ['0', 'ó'],
        'u': ['v', 'ú', 'ü'],
        's': ['$', '5'],
        'l': ['1'],
        't': ['7'],
        'b': ['8'],
        'g': ['9'],
    }

    possible_replacements = []
    for c in word:
        lower_c = c.lower()
        # siempre incluimos el carácter original 'c'
        if lower_c in replacements:
            char_options = [c] + replacements[lower_c]
        else:
            char_options = [c]
        possible_replacements.append(char_options)

    # todas las combinaciones posibles
    all_combinations = list(itertools.product(*possible_replacements))
    variants = [''.join(comb) for comb in all_combinations]

    return variants[:num_variants]


def load_dataset_compact_json(path, tokenizer, max_len, augment=False, num_variants=3):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    labels = []

    for label in data:
        for word in data[label]:
            variants = generate_leetspeak_variants(word, num_variants) if augment else [word]
            for variant in variants:
                texts.append(variant)
                labels.append([int(label)])
    # print(texts)
    tokenizer.build_vocab(texts)

    X = np.array([tokenizer.encode(t, max_len) for t in texts])
    y = np.array(labels)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    return X[indices], y[indices]