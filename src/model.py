import json
import tensorflow as tf
from tensorflow.keras import layers

def load_config(path="configs/config.json"):
    with open(path) as f:
        return json.load(f)

def build_model(vocab_size, config):
    inputs = layers.Input(shape=(config["SEQ_LEN"],), dtype=tf.int32)
    x = layers.Embedding(input_dim=vocab_size, output_dim=config["EMBED_DIM"])(inputs)
    # Must keep return_sequences=True so attention can see all time steps
    lstm_out = layers.LSTM(config["LSTM_UNITS"], return_sequences=True)(x)
    # Self-attention over the LSTM outputs (query=keys=values=lstm_out)
    attn_out = layers.Attention()([lstm_out, lstm_out])
    # Combine original LSTM signal + attended context
    x = layers.Concatenate()([lstm_out, attn_out])
    x = layers.Dropout(config["DROPOUT"])(x)
    # Next-token logits at every timestep
    outputs = layers.Dense(vocab_size)(x)
    lm_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    lm_model.compile(
       optimizer=tf.keras.optimizers.Adam(2e-3),
       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    return lm_model