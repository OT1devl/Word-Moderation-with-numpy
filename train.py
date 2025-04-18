from utils import *
from models import ModerationNetwork
from optimizers import Adam
from settings import *

def main():
    tokenizer = CharTokenizer()

    X_train, y_train = load_dataset_compact_json(words_path, tokenizer, MAX_LEN, augment=True, num_variants=NUM_VARIANTS)
    model = ModerationNetwork(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=128,
        neurons=64
    )
    print(X_train.shape, y_train.shape)
    model.compile(
        optimizer=Adam()
    )
    
    model.train(
        x=X_train,
        y=y_train,
        epochs=2,
        batch_size=32,
        verbose=True,
        print_every=1
    )

    model.save(path=model_path)
    tokenizer.save(path=tokenizer_path)

if __name__ == '__main__':
    main()