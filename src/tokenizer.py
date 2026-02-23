# ── tokenizer.py 
# Vectorizer setup shared between training and inference.
# Training calls build_vectorizer().
# Inference calls restore_vectorizer().

import tensorflow as tf
from tensorflow.keras import layers


def custom_standardize(text):
    """
    Lowercase text, preserve <TAG> style tokens, remove punctuation.
    Must be identical in training and inference — any difference causes
    token ID mismatches and produces garbage output.
    """
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, r"(<[^>]+>)", r" \1 ")
    text = tf.strings.regex_replace(text, r"[^\w\s<>]", " ")
    text = tf.strings.regex_replace(text, r"\s+", " ")
    return tf.strings.strip(text)


def build_vectorizer(text_ds: tf.data.Dataset, vocab_size: int) -> tuple:
    """
    Fit a TextVectorization layer on the training corpus.
    Returns (vectorizer, vocab).
    Called once during training.
    """
    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        standardize=custom_standardize,
        split="whitespace",
        output_mode="int",
    )
    vectorizer.adapt(text_ds.batch(256))
    vocab = vectorizer.get_vocabulary()
    print(f"Vocab size: {len(vocab)}")
    return vectorizer, vocab


def restore_vectorizer(vocab: list) -> layers.TextVectorization:
    """
    Rebuild the vectorizer from a saved vocabulary list.
    vocab[0] = '' (padding), vocab[1] = '[UNK]' — set_vocabulary skips these.
    Called during inference after loading vocab.json from MLflow.
    """
    vectorizer = layers.TextVectorization(
        max_tokens=len(vocab),
        standardize=custom_standardize,
        split="whitespace",
        output_mode="int",
    )
    # Pass vocab[2:] — set_vocabulary reserves index 0 and 1 automatically
    vectorizer.set_vocabulary(vocab[2:])
    return vectorizer