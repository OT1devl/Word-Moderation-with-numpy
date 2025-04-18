import numpy as np

from models import ModerationNetwork
from utils import CharTokenizer
from settings import model_path, tokenizer_path, MAX_LEN

def evaluate_text(tokenizer: CharTokenizer, text: str):
    return np.array([tokenizer.encode(word, max_len=MAX_LEN) for word in text.split()])

def predict_every_word(model: ModerationNetwork, tokens: np.ndarray):
    return np.round(model.forward(tokens), 4)

def main():
    model: ModerationNetwork = ModerationNetwork.load(path=model_path)
    tokenizer: CharTokenizer = CharTokenizer.load(path=tokenizer_path)
    print(tokenizer.idx2char)
    text = "hola que tal estas hij0 de put4"
    evaluation = evaluate_text(tokenizer, text)
    print(evaluation)
    print(predict_every_word(model, evaluation))

if __name__ == '__main__':
    main()