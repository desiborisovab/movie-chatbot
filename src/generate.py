
# Text generation logic — sampling, decoding, and the generation loop.
# Used by chat.py. Has no Spark or MLflow dependencies.

import numpy as np
import tensorflow as tf

from config import SEQ_LEN, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K


def make_fast_predict(lm_model):
    """
    Wrap the model in tf.function for fast token-by-token inference.
    Without this, predict() spins up the full Keras pipeline on every call.
    Call once after loading the model, then pass fast_predict to generate().
    """
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, None), dtype=tf.int32)])
    def fast_predict(x):
        return lm_model(x, training=False)
    return fast_predict


def sample_from_logits(logits, temperature: float = 0.8, top_k: int = 50) -> int:
    """
    Convert raw model logits to a single sampled token ID.

    Steps:
        1. Divide by temperature  (higher = more random, lower = more greedy)
        2. Keep only the top_k highest scoring tokens, mask the rest
        3. Softmax -> probabilities
        4. Sample one token weighted by those probabilities

    Temperature guide:
        0.0 - 0.3  deterministic, repetitive   (debugging)
        0.5        confident, coherent
        0.7        good default
        1.0        raw model distribution
        1.5+       creative but risks incoherence
    """
    logits = tf.cast(logits, tf.float32)
    logits = logits / max(float(temperature), 1e-6)

    if top_k is not None and top_k > 0:
        k_actual = min(int(top_k), tf.shape(logits)[0].numpy())
        values, _ = tf.math.top_k(logits, k=k_actual)
        cutoff    = values[-1]
        logits    = tf.where(
            logits < cutoff,
            tf.constant(-1e10, dtype=logits.dtype),
            logits,
        )

    probs   = tf.nn.softmax(logits)
    next_id = int(tf.random.categorical(tf.math.log([probs]), 1)[0, 0])
    return next_id


def detokenize(ids: list, id_to_token: np.ndarray) -> str:
    """Convert a list of token IDs back to a readable string."""
    toks = id_to_token[ids]
    toks = [t for t in toks if t not in ("", "[UNK]")]
    return " ".join(toks)


def generate(
    prompt:         str,
    vectorizer,
    fast_predict,
    id_to_token:    np.ndarray,
    max_new_tokens: int   = DEFAULT_MAX_NEW_TOKENS,
    temperature:    float = DEFAULT_TEMPERATURE,
    top_k:          int   = DEFAULT_TOP_K,
) -> str:
    """
    Generate text continuation for a given prompt.

    Args:
        prompt:         formatted movie prompt string
        vectorizer:     restored TextVectorization layer
        fast_predict:   tf.function wrapped model (from make_fast_predict)
        id_to_token:    numpy array mapping token ID -> word
        max_new_tokens: number of tokens to generate
        temperature:    sampling temperature
        top_k:          vocabulary to sample from at each step

    Returns:
        Generated text string (prompt not included)
    """
    ids = vectorizer(tf.constant([prompt]))[0]
    ids = tf.boolean_mask(ids, ids > 0).numpy().tolist()

    prompt_len = len(ids)   # mark boundary — return only generated part

    for _ in range(max_new_tokens):
        # take the last SEQ_LEN tokens as context
        window = ids[-SEQ_LEN:]
        if len(window) < SEQ_LEN:
            window = [0] * (SEQ_LEN - len(window)) + window   # left-pad

        x       = tf.constant([window], dtype=tf.int32)
        logits  = fast_predict(x)                              # (1, SEQ_LEN, vocab_size)
        next_id = sample_from_logits(logits[0, -1], temperature=temperature, top_k=top_k)
        ids.append(next_id)

    return detokenize(ids[prompt_len:], id_to_token)