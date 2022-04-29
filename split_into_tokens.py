from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

def tokenize(ds_name, tokenizer_savedir, max_len, vocab_size):
    
    special_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]

    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    trainer = WordLevelTrainer(show_progress = True,
                           special_tokens = special_tokens,
                           vocab_size = vocab_size,
                           padding="max_length",
                           max_length=max_len)

    tokenizer.pre_tokenizer = Whitespace()

    file_paths = [str(x) for x in Path(ds_name).glob("*.txt")]
    tokenizer.train(file_paths, trainer)

    tokenizer.save(f"{tokenizer_savedir}/tokenizer.json")

if __name__ == "__main__":
    
    vocab_size = 128

    ds_name = f"gs_{vocab_size}c/gs_train"
    tokenizer_savedir = f"gs_{vocab_size}c/gs_{vocab_size}c_tokenizer"
    max_len = 1292
    tokenize(ds_name, tokenizer_savedir, max_len, vocab_size)
