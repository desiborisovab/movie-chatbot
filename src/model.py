# Entry point for training.
# Reads from:  config.py, data.py, tokenizer.py, model.py
# Writes to:MLflow (model, vocab, vectorizer weights, params, metrics)

import json
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from config   import *
from data     import load_docs
from tokenizer import build_vectorizer
from model    import build_model

# Load data
docs    = load_docs(spark, MOVIE_TABLE)
text_ds = tf.data.Dataset.from_tensor_slices(docs).shuffle(10_000, seed=42)

# Build tokenizer 
vectorizer, vocab = build_vectorizer(text_ds, VOCAB_SIZE)
vocab_size        = len(vocab)

# Build dataset pipeline
def make_windows(token_ids):
    return (
        tf.data.Dataset.from_tensor_slices(token_ids)
        .window(SEQ_LEN + 1, shift=STRIDE, drop_remainder=True)
        .flat_map(lambda w: w.batch(SEQ_LEN + 1))
    )

def split_xy(seq):
    return seq[:-1], seq[1:]

def doc_to_ds(doc):
    ids = vectorizer(tf.expand_dims(doc, 0))[0]
    ids = tf.boolean_mask(ids, ids > 0)
    return make_windows(ids).map(split_xy, num_parallel_calls=tf.data.AUTOTUNE)

lm_ds = text_ds.flat_map(doc_to_ds)

val_ds = (
    lm_ds.take(VAL_EXAMPLES)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

train_ds_rep = (
    lm_ds.skip(VAL_EXAMPLES)
    .repeat()
    .shuffle(20_000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds_rep = val_ds.repeat()

# Build and compile model 
lm_model = build_model(
    vocab_size=vocab_size,
    seq_len=SEQ_LEN,
    embed_dim=EMBED_DIM,
    lstm_units=LSTM_UNITS,
)
lm_model.summary()

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=STEPS_PER_EPOCH * EPOCHS,
    alpha=1e-6,
)

lm_model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# ── 5. Train 
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
]

history = lm_model.fit(
    train_ds_rep,
    validation_data=val_ds_rep,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Log to MLflow 
mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as run:
    run_id = run.info.run_id

    mlflow.tensorflow.log_model(lm_model, artifact_path="model")
    mlflow.log_text(json.dumps(vocab),                                  "vocab.json")
    mlflow.log_text(json.dumps([w.tolist() for w in vectorizer.get_weights()]),
                                                                        "vectorizer_weights.json")

    mlflow.log_params({
        "SEQ_LEN":              SEQ_LEN,
        "VOCAB_SIZE":           vocab_size,
        "BATCH_SIZE":           BATCH_SIZE,
        "STRIDE":               STRIDE,
        "STEPS_PER_EPOCH":      STEPS_PER_EPOCH,
        "VAL_STEPS":            VAL_STEPS,
        "EPOCHS":               EPOCHS,
        "LEARNING_RATE":        LEARNING_RATE,
        "EARLY_STOP_PATIENCE":  EARLY_STOP_PATIENCE,
        "EMBED_DIM":            EMBED_DIM,
        "LSTM_UNITS":           LSTM_UNITS,
    })

    for epoch, (train_loss, val_loss) in enumerate(
        zip(history.history["loss"], history.history["val_loss"])
    ):
        mlflow.log_metrics(
            {"train_loss": train_loss, "val_loss": val_loss},
            step=epoch,
        )

print("Training complete. run_id =", run_id)