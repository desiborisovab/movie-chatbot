# src/inference.py
import json
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from src.retrieval import retrieve_movies, format_context
from src.tokenizer import build_vectorizer  # <-- critical: same standardize as training


def load_artifacts_from_mlflow(run_id: str):
    """
    Loads:
      - Keras model from MLflow run artifacts
      - vocab.json from MLflow run artifacts
      - SEQ_LEN from MLflow run params
      - Reconstructs TextVectorization with the SAME standardization as training
    """
    # Load model
    model = mlflow.tensorflow.load_model(f"runs:/{run_id}/model")

    # Load vocab artifact (stored as JSON list)
    vocab_text = mlflow.artifacts.load_text(f"runs:/{run_id}/vocab.json")
    vocab = json.loads(vocab_text)  # list[str]

    # Load SEQ_LEN from run params
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run_id).data.params
    seq_len = int(params["SEQ_LEN"])

    # Rebuild vectorizer EXACTLY like training
    # (same standardize function, same split, etc.)
    vectorizer = build_vectorizer(vocab_size=len(vocab), seq_len=seq_len)
    vectorizer.set_vocabulary(vocab)

    id_to_token = np.array(vocab)

    return model, vectorizer, vocab, id_to_token, seq_len


def sample_from_logits(logits, temperature: float = 0.5, top_k: int = 5su) -> int:
    """
    Sample a token id from a logits vector.

    temperature:
      - < 1.0 makes output more deterministic
      - > 1.0 makes output more random

    top_k:
      - keeps only the top_k candidate tokens (prevents low-probability garbage)
    """
    logits = tf.cast(logits, tf.float32)
    logits = logits / max(float(temperature), 1e-6)

    if top_k is not None and top_k > 0:
        k = min(int(top_k), int(logits.shape[-1]))
        values, _ = tf.math.top_k(logits, k=k)
        cutoff = values[-1]
        logits = tf.where(logits < cutoff, tf.constant(-1e10, logits.dtype), logits)

    probs = tf.nn.softmax(logits)
    next_id = int(tf.random.categorical(tf.math.log([probs]), 1)[0, 0])
    return next_id


def detokenize(ids, id_to_token) -> str:
    toks = id_to_token[ids]
    toks = [t for t in toks if t not in ("", "[UNK]")]
    return " ".join(toks)


def generate(
    model,
    vectorizer,
    id_to_token,
    seq_len: int,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_k: int = 50,
) -> str:
    """
    Autoregressive generation:
      1) tokenize prompt -> ids
      2) loop:
           - take last seq_len ids as the window
           - model predicts logits for each timestep
           - take logits at last timestep and sample next token
           - append next token id
      3) detokenize all ids to text
    """
    # Vectorize prompt -> (seq_len,) because output_sequence_length=seq_len
    # But we still remove zeros to build a growing history
    ids = vectorizer(tf.constant([prompt]))[0]
    ids = tf.boolean_mask(ids, ids > 0).numpy().tolist()

    for _ in range(max_new_tokens):
        window = ids[-seq_len:]
        if len(window) < seq_len:
            window = [0] * (seq_len - len(window)) + window

        x = tf.constant([window], dtype=tf.int32)  # (1, seq_len)
        logits = model(x)                          # (1, seq_len, vocab_size)
        next_id = sample_from_logits(logits[0, -1], temperature=temperature, top_k=top_k)
        ids.append(next_id)

    return detokenize(ids, id_to_token)


class MovieRAGBot:
    """
    Holds the model + vectorizer in memory so you don't reload MLflow on every call.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.model, self.vectorizer, self.vocab, self.id_to_token, self.seq_len = load_artifacts_from_mlflow(run_id)

    def answer(
        self,
        spark,
        question: str,
        k: int = 5,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        top_k: int = 50,
        movie_table: str = "default.wiki_movie_plots_deduped",
        use_tags: bool = True,
    ) -> str:
        rows = retrieve_movies(spark, question, k=k, movie_table=movie_table).collect()

        # Important: format_context should match how you trained (tags vs plain labels).
        # If you trained with <TITLE>, <YEAR>, ... then keep them here too.
        context = format_context(rows, use_tags=use_tags)

        prompt = (
            "You are a movie assistant.\n"
            "Use only the CONTEXT to answer. If the answer is not in the context, say \"I don't know.\"\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER:\n"
        )

        return generate(
            model=self.model,
            vectorizer=self.vectorizer,
            id_to_token=self.id_to_token,
            seq_len=self.seq_len,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )


def answer(
    spark,
    run_id: str,
    question: str,
    k: int = 5,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_k: int = 50,
    movie_table: str = "default.wiki_movie_plots_deduped",
    use_tags: bool = True,
) -> str:
    """
    Convenience function (simple API).
    For repeated calls, prefer MovieRAGBot(run_id).answer(...) to avoid reloading artifacts.
    """
    bot = MovieRAGBot(run_id)
    return bot.answer(
        spark=spark,
        question=question,
        k=k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        movie_table=movie_table,
        use_tags=use_tags,
    )
