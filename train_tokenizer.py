from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_bpe_tokenizer(corpus_files, special_tokens, continuing_subword_prefix="##", min_frequency=1,vocab_size=30000):
    """

    This function initializes a tokenizer, specified vocabulary size and list of special tokens.
    The tokenizer is then trained on the provided corpus files.

    Args:
        corpus_files (list of str): A list of paths to the text files used for training the tokenizer.
        continuing_subword_prefix (str, optional):  An indcator of continuing prefix.
        min_frequency (integer, optional):  A parameter to control vocabulary size 
        special_tokens (list of str): A list of special tokens to include in the tokenizer's vocabulary.
        vocab_size (int, optional): The maximum size of the vocabulary. Defaults to 30,000.

    Returns:
        Tokenizer: The trained BPE tokenizer.

    Example usage:
        >>> corpus_files = ["./data/text1.txt", "./data/text2.txt"]
        >>> special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", ....]
        >>> tokenizer = train_bpe_tokenizer(corpus_files, special_tokens)
    """
    # intialize the tokenizer, here i have bpe tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # I first defined a pre-tokenizer which splits on whitespace
    tokenizer.pre_tokenizer = Whitespace()
    # here trainer with the desired vocabulary size and special tokens
    trainer = BpeTrainer(continuing_subword_prefix=continuing_subword_prefix, min_frequency=min_frequency, vocab_size=vocab_size, special_tokens=special_tokens)
    # training
    tokenizer.train(files=corpus_files, trainer=trainer)
    return tokenizer

# path to your all training data or the data you want to train your tokenizer on it.
corpus_files = ["requirement.txt"]

# train the tokenizer
# make sure that you add any special tokens that you may include in your preprocessing steps, for instance i added profanity 
bpe_tokenizer = train_bpe_tokenizer(corpus_files, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[PROFANITY]"])

# save the tokenizer
bpe_tokenizer.save("trained_bpe_tokenizer.json")

# it's essential to see how you words have been tokenized
"""
<<<<<<< HEAD
encoded_output = bpe_tokenizer.encode("Hello, world!")
print(bpe_tokenizer.get_vocab())
=======
encoded_output = bpe_tokenizer.encode("hello, world!")
print(encoded_output.tokens)
>>>>>>> 659421c7f91bdbab61f9fabda6e20975778c17fe
"""
