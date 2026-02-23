# Inference entry point. Run this notebook in Databricks after training.
# Loads the model and vocab from MLflow, then exposes chat() for interactive use.
#
# Usage:
#   RUN_ID = "your_run_id_here"
#   print(chat("dark knight batman gotham"))

import json
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from config    import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K
from tokenizer import restore_vectorizer
from generate  import make_fast_predict, generate
from retrieve  import retrieve_movies, row_to_prompt

# Set your run ID 
RUN_ID = "cf085433cd7f4585bed5366959298606"   # ← update after each training run

# Load model from MLflow 
print("Loading model...")
lm_model = mlflow.tensorflow.load_model(f"runs:/{RUN_ID}/model")

client  = mlflow.tracking.MlflowClient()
params  = client.get_run(RUN_ID).data.params
SEQ_LEN = int(params["SEQ_LEN"])
print(f"SEQ_LEN: {SEQ_LEN}")

# Load vocab and restore vectorizer 
print("Restoring vectorizer...")
vocab       = json.loads(mlflow.artifacts.load_text(f"runs:/{RUN_ID}/vocab.json"))
id_to_token = np.array(vocab)
vectorizer  = restore_vectorizer(vocab)

# sanity check — should print real tokens, no [UNK]
test_ids    = vectorizer(tf.constant(["<TITLE> batman <PLOT> fights the joker"]))[0].numpy()
test_tokens = [vocab[i] for i in test_ids if i > 0]
print("Sanity check:", test_tokens)

# ── 4. Compile fast inference function 
fast_predict = make_fast_predict(lm_model)
print("Model ready.\n")

# ── 5. Chat function 
def chat(
    question:       str,
    max_new_tokens: int   = DEFAULT_MAX_NEW_TOKENS,
    temperature:    float = DEFAULT_TEMPERATURE,
    top_k:          int   = DEFAULT_TOP_K,
) -> str:
    """
    Retrieve the best matching movie for the question,
    use it as a prompt, and generate a continuation.

    Args:
        question:       natural language query, e.g. "dark knight batman"
        max_new_tokens: number of tokens to generate
        temperature:    0.7 is a good default (see generate.py for guide)
        top_k:          vocabulary size at each sampling step

    Returns:
        Generated text string
    """
    hits = retrieve_movies(question, spark, k=3).toPandas()

    if hits.empty:
        return "No matching movies found."

    top    = hits.iloc[0]
    prompt = row_to_prompt(top)

    print(f"[Retrieved: {top['Title']} ({top.get('Release Year', '?')})]")
    print(f"[Prompt: {prompt[:120]}...]\n")

    return generate(
        prompt         = prompt,
        vectorizer     = vectorizer,
        fast_predict   = fast_predict,
        id_to_token    = id_to_token,
        max_new_tokens = max_new_tokens,
        temperature    = temperature,
        top_k          = top_k,
    )


# Try it 
# print(chat("dark knight batman gotham"))
# print(chat("romantic comedy paris 1990s"))
# print(chat("science fiction space war"))