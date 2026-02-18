# src/data_prep.py
# Data preparation for the Movie Chatbot language model:
# - Reads Spark table(s)
# - Cleans + formats each row into a structured "doc" string
# - Builds TextVectorization vocabulary
# - Converts docs -> token ids -> fixed-length windows
# - Produces tf.data datasets for next-token prediction (X, y)

import re
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers


# Text cleaning + formatting

def clean_text(s) -> str:
    """
    Normalize whitespace and handle missing values.
    - None -> ""
    - Replace non-breaking space with normal space
    - Collapse repeated whitespace/newlines/tabs
    """
    if s is None:
        return ""
    s = str(s).replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_year(y) -> str:
    """
    Convert year-like field to a clean integer string.
    If conversion fails, return empty string.
    """
    try:
        return str(int(y))
    except Exception:
        return ""


def take_cast(cast: str, n: int = 5) -> str:
    """
    Keep only the first N cast members (comma-separated list).
    """
    cast = clean_text(cast)
    if not cast:
        return ""
    parts = [p.strip() for p in cast.split(",") if p.strip()]
    return ", ".join(parts[:n])


def row_to_doc(r, config: dict) -> str:
    """
    Convert one wiki_movie_plots_deduped row to a structured text block.

    We add "field tokens" like <TITLE>, <PLOT> to make the format consistent.
    That helps the model learn structure and improves generation stability.
    """
    title    = clean_text(r["Title"])
    year     = safe_year(r["Release Year"])
    origin   = clean_text(r["Origin/Ethnicity"])
    director = clean_text(r["Director"])
    genre    = clean_text(r["Genre"])
    cast     = take_cast(r["Cast"], n=config.get("CAST_TOP_N", 5))
    plot     = clean_text(r["Plot"])

    # Skip if the row is missing core info
    if not title or not plot:
        return ""

    return (
        f"<TITLE> {title}\n"
        f"<YEAR> {year}\n"
        f"<ORIGIN> {origin}\n"
        f"<DIRECTOR> {director}\n"
        f"<CAST> {cast}\n"
        f"<GENRE> {genre}\n"
        f"<PLOT> {plot}\n"
    ).strip()


def build_docs_from_wiki(spark, config: dict) -> List[str]:
    """
    Read the wiki movie table into pandas and convert each row into one doc string.
    Note: toPandas() brings all rows to the driver. This is OK for learning/projects,
    but for "big data" you would do this with Spark transformations instead.
    """
    table = config.get("WIKI_TABLE", "default.wiki_movie_plots_deduped")
    df = spark.table(table).toPandas()

    docs = [row_to_doc(r, config) for _, r in df.iterrows()]
    docs = [d for d in docs if d]
    return docs


# Tokenization / Vectorizer

def custom_standardize(text: tf.Tensor) -> tf.Tensor:
    """
    Custom standardization that preserves <TITLE> style tokens.

    Steps:
    1) lowercase
    2) ensure tokens like <TITLE> are surrounded by spaces (so they become their own tokens)
    3) remove punctuation except < and >
    4) collapse whitespace
    """
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, r"(<[^>]+>)", r" \1 ")
    text = tf.strings.regex_replace(text, r"[^\w\s<>]", " ")
    text = tf.strings.regex_replace(text, r"\s+", " ")
    return tf.strings.strip(text)


def build_vectorizer(docs: List[str], config: dict) -> Tuple[layers.TextVectorization, List[str]]:
    """
    Create and adapt the TextVectorization layer.
    Returns:
      vectorizer: fitted tokenizer layer
      vocab: list of tokens (index = token id)
    """
    vocab_size = int(config["VOCAB_SIZE"])

    text_ds = tf.data.Dataset.from_tensor_slices(docs)

    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        standardize=custom_standardize,
        split="whitespace",
        output_mode="int",
    )

    # adapt() expects a dataset of batches
    vectorizer.adapt(text_ds.batch(int(config.get("ADAPT_BATCH", 256))))
    vocab = vectorizer.get_vocabulary()

    return vectorizer, vocab



# Windowing to (X, y)


def make_windows(token_ids: tf.Tensor, seq_len: int) -> tf.data.Dataset:
    """
    Convert 1D token_ids -> dataset of windows length (seq_len+1).

    Example:
    token_ids: [10, 11, 12, 13, 14, 15, ...]
    seq_len=4 => windows length 5:
      [10,11,12,13,14]
      [14,15,16,17,18]   (shift = seq_len)
    """
    return (
        tf.data.Dataset.from_tensor_slices(token_ids)
        .window(seq_len + 1, shift=seq_len, drop_remainder=True)
        .flat_map(lambda w: w.batch(seq_len + 1))
    )


def split_xy(seq: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Convert a window of length (seq_len+1) into:
    X = first seq_len tokens
    y = next seq_len tokens (shifted by 1)
    """
    return seq[:-1], seq[1:]


def doc_to_lm_examples(doc: tf.Tensor, vectorizer: layers.TextVectorization, seq_len: int) -> tf.data.Dataset:
    """
    doc (string) -> token ids -> remove padding zeros -> windows -> (X,y)
    """
    ids = vectorizer(tf.expand_dims(doc, 0))[0]        # (tokens,)
    ids = tf.boolean_mask(ids, ids > 0)               # drop zeros
    return make_windows(ids, seq_len).map(split_xy, num_parallel_calls=tf.data.AUTOTUNE)


def prepare_datasets(spark, config: dict):
    """
    Main entry point used by train.py

    Returns:
      train_ds_rep: repeated training dataset (for Keras fit with steps_per_epoch)
      val_ds_rep: repeated validation dataset
      vocab: list[str]
      vectorizer: TextVectorization layer (optional but useful for debugging/inference)
    """
    docs = build_docs_from_wiki(spark, config)

    seq_len = int(config["SEQ_LEN"])
    batch_size = int(config["BATCH_SIZE"])

    # Build vectorizer + vocab
    vectorizer, vocab = build_vectorizer(docs, config)

    # Dataset of docs (strings)
    text_ds = (
        tf.data.Dataset.from_tensor_slices(docs)
        .shuffle(int(config.get("DOC_SHUFFLE", 10000)), seed=int(config.get("SEED", 42)))
    )

    # Expand each doc to many (X,y) windows, then shuffle examples
    lm_ds = (
        text_ds
        .flat_map(lambda d: doc_to_lm_examples(d, vectorizer, seq_len))
        .shuffle(int(config.get("EXAMPLE_SHUFFLE", 20000)), seed=int(config.get("SEED", 42)))
    )

    # Split
    val_examples = int(config.get("VAL_EXAMPLES", 5000))

    val_ds = (
        lm_ds.take(val_examples)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    train_ds = (
        lm_ds.skip(val_examples)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Repeat so Keras can run for a fixed number of steps per epoch
    train_ds_rep = train_ds.repeat()
    val_ds_rep = val_ds.repeat()

    return train_ds_rep, val_ds_rep, vocab, vectorizer
